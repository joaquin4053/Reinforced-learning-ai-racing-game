"""Automatische Tests für die Simulation (Spiel v2, 3D).   Ausführen:  .venv/bin/python -m pytest"""
import math
import random
import re
from pathlib import Path

import numpy as np
import pytest

from game import sensors
from game.sim import Rules, Sim, action
from game.track import Track
from tools.autopilot import autopilot

TRACKS = sorted(Path("tracks").glob("*.json"))
ENDLESS = Rules(stuck_seconds=1e9, wrong_way_meters=1e9, time_limit=1e9)


def random_inputs(n, seed):
    rng = random.Random(seed)
    return [action(rng.uniform(-1, 1), rng.random(), rng.random() * 0.3, float(rng.random() < 0.1)) for _ in range(n)]


@pytest.mark.parametrize("path", TRACKS, ids=lambda p: p.stem)
def test_track_loads_and_is_drivable(path):
    track = Track.load(path)
    sim = Sim(track)
    while not sim.done:
        sim.step(autopilot(sim, 40))          # vorsichtiger Testfahrer (144 km/h, Lücken brauchen Mindesttempo)
    assert sim.result == "finished", f"{path.stem}: {sim.end_reason} bei {sim.progress:.0f} m"
    assert len(sim.splits) == len(track.checkpoints) + 1


def test_deterministic():
    track = Track.load(TRACKS[0])
    runs = []
    for _ in range(2):
        sim = Sim(track, rules=ENDLESS)
        for a in random_inputs(1500, seed=42):
            sim.step(a)
        runs.append(sim.history)
    assert runs[0] == runs[1]


def _straight(rand):
    return Track({"name": "gerade", "width": 28, "rand": rand, "start": [0, 30, 0, 0],
                  "pieces": [["start"], ["straight", 4000], ["finish"], ["straight", 30]]})


def _hit_border(speed, angle_deg):
    """Schräg auf den Streckenrand zu fahren. True = hinausgeflogen."""
    sim = Sim(_straight(True), rules=Rules(stuck_seconds=1e9, time_limit=1e9, wrong_way_meters=1e9))
    c = sim.car
    c.heading = math.radians(angle_deg)
    c.vx, c.vz = math.sin(c.heading) * speed, math.cos(c.heading) * speed
    for _ in range(60 * 12):
        vf, _ = c.local_velocity()
        sim.step(action(throttle=0.5 if vf < speed else 0.0))
        if sim.done:
            break
    return sim.end_reason == "fell"


def test_border_holds_unless_far_too_fast():
    assert not _hit_border(11, 45)          # 40 km/h, steil: der Rand hält
    assert not _hit_border(55, 10)          # 200 km/h, flach: hält
    assert _hit_border(90, 30)              # 324 km/h, steil: fliegt drüber


def test_driving_off_the_edge_is_a_crash():
    track = _straight(False)                # ohne Rand
    sim = Sim(track)
    while not sim.done:
        sim.step(action(steer=0.6, throttle=1.0))   # lenkt dauernd nach rechts -> fällt runter
    assert sim.result == "crashed" and sim.end_reason == "fell"
    assert sim.finish_time is None


def test_gates_in_order_only():
    track = Track.load(TRACKS[0])
    sim = Sim(track)
    g = track.finish_gate                              # Ziel direkt überqueren zählt nicht
    assert not sim.gates[sim.next_gate] is g
    assert sim.next_gate == 0


def test_actions_are_clipped():
    track = Track.load(TRACKS[0])
    a, b = Sim(track, rules=ENDLESS), Sim(track, rules=ENDLESS)
    for _ in range(200):
        a.step([5.0, 9.0, -3.0, 7.0])
        b.step([1.0, 1.0, 0.0, 1.0])
    assert a.history == b.history


def test_observation_matches_claude_md():
    track = Track.load(TRACKS[0])
    obs = Sim(track).observation()
    assert obs.shape == (sensors.SIZE,) and obs.dtype == np.float32
    assert np.all(obs >= sensors.LOW) and np.all(obs <= sensors.HIGH)
    text = Path("CLAUDE.md").read_text()
    names = re.findall(r"^\| \d+ \| `(\w+)` \|", text, flags=re.M)
    assert names == [row[0] for row in sensors.LAYOUT], "Tabelle in CLAUDE.md passt nicht zu game/sensors.py"


def test_jump_flies_and_lands():
    track = Track({"name": "rampe", "width": 22, "start": [0, 60, 0, 0],
                   "pieces": [["start"], ["straight", 300], ["ramp", 30, 4], ["gap", 40, -6], ["slope", 60, -4],
                              ["straight", 300], ["finish"], ["straight", 30]]})
    sim = Sim(track)
    took_off = landed = False
    while not sim.done:
        ev = sim.step(autopilot(sim, 45))
        took_off |= ev["takeoff"]
        landed |= ev["landed"] > 0
    assert took_off and landed and sim.result == "finished"


def test_rays_see_the_edge():
    track = _straight(False)
    sim = Sim(track)
    rays = sim.rays()
    left, right = rays[0], rays[-1]                      # 90° links/rechts: halbe Fahrbahnbreite
    assert abs(left - track.width / 2) < 1.0 and abs(right - track.width / 2) < 1.0
    walled = Sim(_straight(True)).rays()                 # mit Rand: der steile Teil zählt als Begrenzung
    assert track.width / 2 - 4 < walled[0] <= track.width / 2
    assert rays[len(rays) // 2] > 50                     # geradeaus: lange Gerade
    assert math.isclose(sensors.RAY_ANGLES_DEG[0], -90)


def test_inner_line_of_banked_turn_is_not_a_crash():
    """Fehler 2026-10-18: In breiten Steilkurven liegt die Innenseite viele Meter tiefer als die Mitte.
    Wer dort fuhr, galt fälschlich als "runtergefallen"."""
    track = Track({"name": "steil", "width": 50, "rand": True, "start": [0, 40, 0, 0],
                   "pieces": [["start"], ["straight", 60], ["turn", 180, 90, 30], ["straight", 120], ["finish"],
                              ["straight", 30]]})
    sim = Sim(track)
    c = sim.car
    while not sim.done:
        (tx, tz), (dx, dz) = track.point_at(sim.progress + 15)
        tx, tz = tx + dz * 16, tz - dx * 16                 # 16 m rechts der Mitte = Innenseite (Rechtskurve)
        ang = (math.atan2(tx - c.x, tz - c.z) - c.heading + math.pi) % (2 * math.pi) - math.pi
        vf, _ = c.local_velocity()
        sim.step(action(steer=max(-1, min(1, ang * 3)), throttle=1.0 if vf < 22 else 0.0))
    assert sim.result == "finished", f"{sim.end_reason} bei {sim.progress:.0f} m, seitlich {sim.lateral:+.1f} m"
