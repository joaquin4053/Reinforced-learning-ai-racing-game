"""Simulation ohne Grafik: Auto + Strecke + Regeln + Sensoren.

Benutzung (gleich für Spieler, KI und Tests):
    sim = Sim(Track.load("tracks/map01.json"))
    obs = sim.reset()
    events = sim.step({"steer": 0.0, "throttle": 1.0, "brake": 0.0})
    if sim.done: ...

Deterministisch: kein Zufall, fester Zeitschritt. Gleiche Aktionen -> gleiche Fahrt.
"""
from dataclasses import dataclass

import numpy as np

from .car import DT, Car
from .raycast import cast_rays, closest_wall

# Erhöhen, sobald sich Physik oder Regeln ändern -> alte Geister werden als "alte Version" markiert
PHYSICS_VERSION = 6  # 2: Wand kein Crash, 3: Reifenmodell, 4: Grip-Abriss + Motorbremse,
                     # 5: Feinschliff, 6: kein Übersteuern mehr

# Erhöhen, sobald sich ändert, was die KI "sieht" (observation, Strahlen, Reichweite)
SENSOR_VERSION = 1

RAY_ANGLES = np.radians(np.linspace(-90, 90, 11))  # Sensoren relativ zur Fahrtrichtung
RAY_RANGE = 100.0
SPEED_NORM = 50.0
COLLISION_CIRCLES = (1.1, -1.1)  # zwei Kreise entlang der Längsachse (vorne/hinten)
COLLISION_RADIUS = 1.05
RESTITUTION = 0.25               # wie stark das Auto von der Wand abprallt
WALL_FRICTION = 0.97             # Tempoverlust pro Schritt an der Wand

END_REASONS = {
    "wall": "Wand berührt",
    "stuck": "Kein Fortschritt",
    "wrong_way": "Falsche Richtung",
    "time_limit": "Zeitlimit",
}


@dataclass
class Rules:
    """Wann endet eine Fahrt als Crash? Gilt für Spieler und KI gleich."""
    wall_is_crash: bool = False      # False: abprallen und weiterfahren (wird nur gemeldet)
    stuck_seconds: float = 2.0       # ...wenn in so vielen Sekunden
    stuck_min_progress: float = 1.0  # ...weniger als so viele Meter Fortschritt
    wrong_way_meters: float = 15.0   # so weit hinter den eigenen Bestwert zurück = falsche Richtung
    time_limit: float = 180.0        # Sekunden


def action(steer=0.0, throttle=0.0, brake=0.0, drift=0.0):
    return {"steer": steer, "throttle": throttle, "brake": brake, "drift": drift}


class Sim:
    def __init__(self, track, rules=None, record=True):
        self.track = track
        self.rules = rules or Rules()
        self.record = record
        self.gates = track.checkpoints + [track.finish_gate]  # Reihenfolge einer Runde
        self.reset()

    # ---------- Ablauf ----------
    def reset(self):
        t = self.track
        self.car = Car(t.start_pos[0], t.start_pos[1], t.start_heading)
        self.frame = 0
        self.lap = 0                 # fertige Runden
        self.next_gate = 0           # Index in self.gates
        self.splits = []             # Zeit bei jedem durchfahrenen Checkpoint/Ziel
        self.done = False
        self.result = None           # "finished" oder "crashed"
        self.end_reason = None       # Schlüssel aus END_REASONS
        self.finish_time = None
        self.wall_contact = False    # berührt das Auto gerade eine Wand?
        self.wall_hits = 0           # wie oft die Wand neu berührt wurde (für Strafe/Statistik)
        self.wall_frames = 0         # wie viele Schritte das Auto an der Wand war
        dist, self._seg, self.track_dir, self.lateral = t.progress_at(*t.start_pos)
        self._raw_dist = dist
        self.progress = dist         # Meter entlang der Strecke, läuft über Runden weiter
        self.best_progress = dist
        self._progress_log = [dist]
        self.history = [self._pose()] if self.record else []
        self._rays = None
        return self.observation()

    @property
    def time(self):
        return self.frame * DT

    @property
    def total_length(self):
        return self.track.length * self.track.laps

    @property
    def progress_pct(self):
        return 100.0 if self.result == "finished" else max(0.0, min(100.0, 100 * self.progress / self.total_length))

    def step(self, act):
        """Einen Zeitschritt (1/60 s) simulieren.

        act: {"steer": -1..1, "throttle": 0..1, "brake": 0..1, "drift": 0..1 (optional)}
        Rückgabe: Events dieses Schritts, z.B. {"checkpoint": "CP 2", "split": 12.35}
        """
        events = {"checkpoint": None, "split": None, "lap": False, "finished": False, "crashed": False,
                  "wall": False, "wall_hit": False}
        if self.done:
            return events
        x0, z0 = self.car.x, self.car.z
        self.car.update(act.get("steer", 0.0), act.get("throttle", 0.0), act.get("brake", 0.0),
                        act.get("drift", 0.0))
        touching = self._collide()
        # "wall" = Auto ist an der Wand, "wall_hit" = neue Berührung (vorher frei)
        events["wall"], events["wall_hit"] = touching, touching and not self.wall_contact
        self.wall_contact = touching
        self.wall_frames += touching
        self.wall_hits += events["wall_hit"]
        self.frame += 1
        self._rays = None
        self._update_progress()
        self._check_gates(x0, z0, events)
        if not self.done:
            self._check_rules(events)
        if self.record:
            self.history.append(self._pose())
        return events

    def _check_gates(self, x0, z0, events):
        gate = self.gates[self.next_gate]
        if not gate.crossed(x0, z0, self.car.x, self.car.z):
            return  # nur das NÄCHSTE Tor zählt -> Reihenfolge erzwungen, keine Abkürzungen
        self.splits.append(self.time)
        events["checkpoint"], events["split"] = gate.name, self.time
        self.next_gate += 1
        if self.next_gate == len(self.gates):
            self.next_gate = 0
            self.lap += 1
            events["lap"] = True
            if self.lap >= self.track.laps:
                self._end("finished", None, events)

    def _check_rules(self, events):
        r = self.rules
        stuck_frames = int(r.stuck_seconds / DT)
        reason = None
        if self.wall_contact and r.wall_is_crash:
            reason = "wall"
        elif self.best_progress - self.progress > r.wrong_way_meters:
            reason = "wrong_way"
        elif self.frame >= stuck_frames and \
                self.progress - self._progress_log[-stuck_frames - 1] < r.stuck_min_progress:
            reason = "stuck"
        elif self.time >= r.time_limit:
            reason = "time_limit"
        if reason:
            self._end("crashed", reason, events)

    def _end(self, result, reason, events):
        self.done = True
        self.result, self.end_reason = result, reason
        if result == "finished":
            self.finish_time = self.time
            events["finished"] = True
        else:
            events["crashed"] = True

    # ---------- Physik ----------
    def _collide(self):
        """Auto aus Wänden herausschieben. True, wenn eine Wand berührt wurde."""
        car, walls = self.car, self.track.walls
        hit = False
        fx, fz = car.forward
        for _ in range(3):
            moved = False
            for off in COLLISION_CIRCLES:
                d, nx, nz = closest_wall(car.x + fx * off, car.z + fz * off, walls)
                if d < COLLISION_RADIUS:
                    pen = COLLISION_RADIUS - d
                    car.x += nx * pen
                    car.z += nz * pen
                    vn = car.vx * nx + car.vz * nz
                    if vn < 0:
                        car.vx -= (1 + RESTITUTION) * vn * nx
                        car.vz -= (1 + RESTITUTION) * vn * nz
                    car.vx *= WALL_FRICTION
                    car.vz *= WALL_FRICTION
                    car.yaw_rate *= 0.9          # Wand bremst auch das Drehen
                    hit = moved = True
            if not moved:
                break
        return hit

    def _update_progress(self):
        t = self.track
        dist, self._seg, self.track_dir, self.lateral = t.progress_at(self.car.x, self.car.z, hint=self._seg)
        delta = dist - self._raw_dist
        if t.is_loop:  # über die Start/Ziel-Linie hinweg nicht springen
            if delta > t.length / 2:
                delta -= t.length
            elif delta < -t.length / 2:
                delta += t.length
        self._raw_dist = dist
        self.progress += delta
        self.best_progress = max(self.best_progress, self.progress)
        self._progress_log.append(self.progress)

    def _pose(self):
        c = self.car
        return (c.x, c.z, c.heading)

    # ---------- Sensoren (für die KI) ----------
    def rays(self):
        """Distanzen der Sensorstrahlen in Metern."""
        if self._rays is None:
            c = self.car
            self._rays = cast_rays(c.x, c.z, c.heading + RAY_ANGLES, self.track.walls, RAY_RANGE)
        return self._rays

    def observation(self):
        """Was die KI "sieht" (nur Fahrersicht, Regel R5) – 16 Zahlen:
        11 Sensorstrahlen, Tempo vorwärts/seitlich, Lenkradstellung, Winkel zur Strecke (sin/cos).
        Keine Koordinaten, keine Ideallinie, nichts hinter der Sensorreichweite.
        Bei jeder Änderung SENSOR_VERSION erhöhen!
        """
        c = self.car
        vf, vr = c.local_velocity()
        fx, fz = c.forward
        tx, tz = self.track_dir
        extra = [vf / SPEED_NORM, vr / SPEED_NORM, c.steer,
                 fx * tz - fz * tx, fx * tx + fz * tz]
        return np.concatenate([self.rays() / RAY_RANGE, extra]).astype(np.float32)
