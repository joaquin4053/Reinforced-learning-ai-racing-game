"""Simulation ohne Grafik: Auto + Strecke + Regeln + Sensoren (Spiel v2, 3D).

Benutzung (gleich für Spieler, KI und Tests):
    sim = Sim(Track.load("tracks/poly01.json"))
    obs = sim.reset()
    events = sim.step({"steer": 0.0, "throttle": 1.0, "brake": 0.0})   # oder sim.step([0.0, 1.0, 0.0])
    if sim.done: ...

Deterministisch: kein Zufall, fester Zeitschritt. Gleiche Aktionen -> gleiche Fahrt.
Alle Zahlenwerte und Versionsnummern stehen in config.toml.
"""
from dataclasses import dataclass

from . import sensors
from .car import DT, Car
from .config import CONFIG, VERSIONS

PHYSICS_VERSION = VERSIONS["physik"]
RULES_VERSION = VERSIONS["regeln"]
SENSOR_VERSION = VERSIONS["sensoren"]

RAY_ANGLES, RAY_RANGE = sensors.RAY_ANGLES, sensors.RAY_RANGE

END_REASONS = {
    "fell": "Runtergefallen",
    "stuck": "Kein Fortschritt",
    "wrong_way": "Falsche Richtung",
    "time_limit": "Zeitlimit",
}

_RULES = CONFIG["regeln"]


@dataclass
class Rules:
    """Wann endet eine Fahrt als Crash? Gilt für Spieler und KI gleich (Werte aus config.toml)."""
    fall_depth: float = _RULES["fall_depth"]
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
    return f"{d:+.2f} s"


class Sim:
    def __init__(self, track, rules=None, record=True):
        self.track = track
        self.rules = rules or Rules()
        self.record = record
        self.gates = track.checkpoints + [track.finish_gate]
        self.reset()

    # ---------- Ablauf ----------
    def reset(self):
        t = self.track
        self.car = Car(t.start_pos[0], t.start_y, t.start_pos[1], t.start_heading)
        self.frame = 0
        self.lap = 0
        self.next_gate = 0
        self.splits = []
        self.done = False
        self.result = None           # "finished", "crashed" oder "truncated"
        self.end_reason = None       # Schlüssel aus END_REASONS
        self.finish_time = None
        self.wall_contact = False    # gerade gegen eine Blockseite gedrückt?
        self.wall_hits = 0           # Abpraller an Blockseiten
        self.wall_frames = 0
        self.jumps = 0               # wie oft abgehoben
        self.airtime_total = 0.0
        dist, self._seg, self.track_dir, self.lateral = t.progress_at(*t.start_pos)
        self._raw_dist = dist
        self.progress = dist
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
        return self.track.length

    @property
    def progress_pct(self):
        return 100.0 if self.result == "finished" else max(0.0, min(100.0, 100 * self.progress / self.total_length))

    def step(self, act):
        """Einen Zeitschritt (1/60 s) simulieren. Zu grosse Werte werden abgeschnitten (R4).

        Rückgabe: Events dieses Schritts, z.B. {"checkpoint": "CP 2", "split": 12.35, "takeoff": True}
        """
        events = {"checkpoint": None, "split": None, "lap": False, "finished": False, "crashed": False,
                  "truncated": False, "wall": False, "wall_hit": False, "takeoff": False, "landed": 0.0}
        if self.done:
            return events
        c = self.car
        x0, z0, was_grounded = c.x, c.z, c.grounded
        act = as_action(act)
        c.update(act.get("steer", 0.0), act.get("throttle", 0.0), act.get("brake", 0.0),
                 act.get("drift", 0.0), self.track)
        events["wall"], events["wall_hit"] = c.hit, c.hit and not self.wall_contact
        self.wall_contact = c.hit
        self.wall_frames += c.hit
        self.wall_hits += events["wall_hit"]
        events["takeoff"] = was_grounded and not c.grounded
        events["landed"] = c.landed
        self.jumps += events["takeoff"]
        if not c.grounded:
            self.airtime_total += DT
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
            return  # nur das NÄCHSTE Tor zählt -> Reihenfolge erzwungen (R1)
        self.splits.append(self.time)
        events["checkpoint"], events["split"] = gate.name, self.time
        self.next_gate += 1
        if self.next_gate == len(self.gates):
            self.lap = 1
            events["lap"] = True
            self._end("finished", None, events)

    def _check_rules(self, events):
        r, c = self.rules, self.car
        stuck_frames = int(r.stuck_seconds / DT)
        reason = None
        if self._fallen():
            reason = "fell"                                   # von der Strecke gefallen (R2)
        elif self.best_progress - self.progress > r.wrong_way_meters:
            reason = "wrong_way"
        elif self.frame >= stuck_frames and \
                self.progress - self._progress_log[-stuck_frames - 1] < r.stuck_min_progress:
            reason = "stuck"
        if reason:
            self._end("crashed", reason, events)
        elif self.time >= r.time_limit:
            self._end("truncated", "time_limit", events)

    def _fallen(self):
        """Runtergefallen (R2)? Verglichen wird mit der Fahrbahn genau dort, wo das Auto ist (mit Querneigung),
        nicht mit der Mittellinie: In Steilkurven liegt die Innenseite viele Meter tiefer als die Mitte."""
        c, t, depth = self.car, self.track, self.rules.fall_depth
        if c.y < t.min_y - depth:
            return True
        g = t.ground(c.x, c.z)
        if g is not None and c.y > g - depth:                 # Fahrbahn direkt darunter -> nicht gefallen
            return False
        return c.y < t.surface_height_at(self.progress, self.lateral) - depth

    def _end(self, result, reason, events):
        self.done = True
        self.result, self.end_reason = result, reason
        if result == "finished":
            self.finish_time = self.time
            events["finished"] = True
        elif result == "truncated":
            events["truncated"] = True
        else:
            events["crashed"] = True

    def _update_progress(self):
        t = self.track
        dist, self._seg, self.track_dir, self.lateral = t.progress_at(self.car.x, self.car.z, hint=self._seg)
        self.progress += dist - self._raw_dist
        self._raw_dist = dist
        self.best_progress = max(self.best_progress, self.progress)
        self._progress_log.append(self.progress)

    def _pose(self):
        c = self.car
        return (c.x, c.y, c.z, c.heading, c.pitch, c.roll)

    # ---------- Sensoren (für die KI, siehe game/sensors.py) ----------
    def rays(self):
        """Distanzen der Sensorstrahlen in Metern."""
        if self._rays is None:
            c = self.car
            self._rays = sensors.ray_distances(c.x, c.z, c.y, c.heading, self.track)
        return self._rays

    def observation(self):
        """Was die KI "sieht" (nur Fahrersicht, Regel R5), Reihenfolge wie die Tabelle in CLAUDE.md."""
        return sensors.observe(self.car, self.track_dir, self.rays())
