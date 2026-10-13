"""Simulation ohne Grafik: Auto + Strecke + Regeln + Sensoren.

Benutzung (gleich für Spieler, KI und Tests):
    sim = Sim(Track.load("tracks/map01.json"))
    obs = sim.reset()
    events = sim.step({"steer": 0.0, "throttle": 1.0, "brake": 0.0})   # oder sim.step([0.0, 1.0, 0.0])
    if sim.done: ...

Deterministisch: kein Zufall, fester Zeitschritt. Gleiche Aktionen -> gleiche Fahrt.
Alle Zahlenwerte und Versionsnummern stehen in config.toml.
"""
from dataclasses import dataclass

import numpy as np

from .car import DT, Car
from .config import CONFIG, VERSIONS
from . import sensors
from .raycast import closest_wall

# Versionsnummern (Regel R8) – werden in config.toml gepflegt
PHYSICS_VERSION = VERSIONS["physik"]
RULES_VERSION = VERSIONS["regeln"]
SENSOR_VERSION = VERSIONS["sensoren"]

RAY_ANGLES, RAY_RANGE = sensors.RAY_ANGLES, sensors.RAY_RANGE   # Sensoren: siehe game/sensors.py

_COLL = CONFIG["kollision"]
COLLISION_CIRCLES = tuple(_COLL["kreise"])
COLLISION_RADIUS = _COLL["radius"]
RESTITUTION = _COLL["abprall"]
WALL_FRICTION = _COLL["wandreibung"]

END_REASONS = {
    "wall": "Wand berührt",
    "stuck": "Kein Fortschritt",
    "wrong_way": "Falsche Richtung",
    "time_limit": "Zeitlimit",
}

_RULES = CONFIG["regeln"]


@dataclass
class Rules:
    """Wann endet eine Fahrt als Crash? Gilt für Spieler und KI gleich (Werte aus config.toml)."""
    wall_is_crash: bool = _RULES["wall_is_crash"]
    stuck_seconds: float = _RULES["stuck_seconds"]
    stuck_min_progress: float = _RULES["stuck_min_progress"]
    wrong_way_meters: float = _RULES["wrong_way_meters"]
    time_limit: float = _RULES["time_limit"]


ACTION_KEYS = ("steer", "throttle", "brake", "drift")


def action(steer=0.0, throttle=0.0, brake=0.0, drift=0.0):
    return {"steer": steer, "throttle": throttle, "brake": brake, "drift": drift}


def as_action(act):
    """Aktion als Dict oder als Liste [Lenkung, Gas, Bremse, (Drift)] annehmen."""
    if isinstance(act, dict):
        return act
    return dict(zip(ACTION_KEYS, act))


def format_time(t):
    """Zeit anzeigen, z.B. 0:41.65 (Regel R6: auf 0.01 s)."""
    return f"{int(t // 60)}:{t % 60:05.2f}"


def format_delta(d):
    """Differenz zum Geist, z.B. '+0.80 s' (langsamer) oder '-1.25 s' (schneller)."""
    return f"{d:+.2f} s"


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
             oder als Liste [steer, throttle, brake, drift]. Zu große Werte werden abgeschnitten (R4).
        Rückgabe: Events dieses Schritts, z.B. {"checkpoint": "CP 2", "split": 12.35}
        """
        events = {"checkpoint": None, "split": None, "lap": False, "finished": False, "crashed": False,
                  "wall": False, "wall_hit": False}
        if self.done:
            return events
        x0, z0 = self.car.x, self.car.z
        act = as_action(act)
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

    # ---------- Sensoren (für die KI, siehe game/sensors.py) ----------
    def rays(self):
        """Distanzen der Sensorstrahlen in Metern."""
        if self._rays is None:
            c = self.car
            self._rays = sensors.ray_distances(c.x, c.z, c.heading, self.track.walls)
        return self._rays

    def observation(self):
        """Was die KI "sieht" (nur Fahrersicht, Regel R5): Vektor mit sensors.SIZE Zahlen,
        Reihenfolge wie die Tabelle in CLAUDE.md."""
        return sensors.observe(self.car, self.track_dir, self.rays())
