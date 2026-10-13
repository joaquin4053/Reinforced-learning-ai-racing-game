"""Automatische Tests für die Simulation.   Ausführen:  .venv/bin/python -m pytest"""
import random
from pathlib import Path

import numpy as np
import pytest

from game.ghost import Ghost, save_ghost
from game.sim import PHYSICS_VERSION, Rules, Sim, action
from game.track import Track
from tools.autopilot_run import autopilot

TRACKS = sorted(Path("tracks").glob("*.json"))


# Ohne Crash-Regeln läuft die Fahrt bis zum Ende der Eingaben weiter (inkl. Wand-Abprallern)
ENDLESS = Rules(wall_is_crash=False, stuck_seconds=1e9, wrong_way_meters=1e9, time_limit=1e9)


def random_inputs(n, seed):
    """Eine feste, aber "wilde" Eingabefolge (Zufall nur mit Seed)."""
    rng = random.Random(seed)
    return [action(rng.uniform(-1, 1), rng.random(), rng.random() * 0.3) for _ in range(n)]


def drive(sim, inputs):
    sim.reset()
    for a in inputs:
        sim.step(a)
    return list(sim.history), sim.frame, sim.result, sim.splits


@pytest.mark.parametrize("track_file", TRACKS, ids=lambda p: p.stem)
def test_track_loads(track_file):
    """Jede Map in tracks/ lässt sich laden (prüft auch Überlappungen und zu enge Kurven)."""
    t = Track.load(track_file)
    assert t.length > 0 and len(t.walls) > 0


def test_same_inputs_same_drive():
    """Gleiche Eingaben zweimal fahren -> exakt (Bit für Bit) dieselbe Fahrt."""
    track = Track.load(TRACKS[0])
    inputs = random_inputs(3000, seed=42)
    first = drive(Sim(track, rules=ENDLESS), inputs)
    second = drive(Sim(track, rules=ENDLESS), inputs)
    assert first[1] == 3000      # wirklich alle Schritte gefahren
    assert first == second


def test_new_sim_same_as_reset():
    """Eine frische Simulation und eine per reset() wiederverwendete fahren identisch."""
    track = Track.load(TRACKS[0])
    inputs = random_inputs(1500, seed=7)
    sim = Sim(track, rules=ENDLESS)
    drive(sim, random_inputs(500, seed=1))   # erst irgendwas anderes fahren
    assert drive(sim, inputs) == drive(Sim(track, rules=ENDLESS), inputs)


def test_autopilot_finishes_and_ghost_roundtrip(tmp_path):
    """Eine saubere Runde endet mit 'finished', und der Geist enthält genau diese Fahrt."""
    sim = Sim(Track.load(TRACKS[0]))
    while not sim.done:
        sim.step(autopilot(sim, 30))
    assert sim.result == "finished"
    assert len(sim.splits) == len(sim.gates)
    ghost = Ghost(save_ghost(sim, "ai", attempt=5, folder=tmp_path))
    assert ghost.meta["physics_version"] == PHYSICS_VERSION and ghost.compatible
    assert ghost.label == "KI · Versuch 5"
    assert len(ghost.frames) == sim.frame + 1
    x, z, h = ghost.state_at(10**9)            # nach dem Ende bleibt der Geist stehen
    assert abs(x - sim.car.x) < 1e-3 and abs(z - sim.car.z) < 1e-3


def test_wall_contact_does_not_end_run():
    """Wand berühren: Fahrt geht weiter, aber die Berührung wird gemeldet (für die KI-Strafe)."""
    sim = Sim(Track.load(TRACKS[0]))
    hit_events = 0
    while not sim.done:
        events = sim.step(action(steer=1, throttle=1))
        hit_events += events["wall_hit"]
    assert sim.wall_hits >= 1 and hit_events == sim.wall_hits
    assert sim.end_reason != "wall"


def test_wall_can_be_made_fatal():
    """Mit Rules(wall_is_crash=True) beendet die Wand die Fahrt (falls man das mal will)."""
    sim = Sim(Track.load(TRACKS[0]), rules=Rules(wall_is_crash=True))
    while not sim.done:
        sim.step(action(steer=1, throttle=1))
    assert sim.result == "crashed" and sim.end_reason == "wall"


def test_standing_still_ends_run():
    sim = Sim(Track.load(TRACKS[0]))
    while not sim.done:
        sim.step(action())
    assert sim.end_reason == "stuck"
    assert sim.time == pytest.approx(sim.rules.stuck_seconds, abs=0.02)


def test_finish_line_without_checkpoints_does_not_count():
    track = Track.load(TRACKS[0])
    sim = Sim(track)
    g = track.finish_gate
    sim.car.x, sim.car.z = g.center - g.tangent * 3     # kurz vor die Ziellinie "beamen"
    for _ in range(30):
        sim.step(action(throttle=1))
    assert sim.splits == [] and sim.result != "finished"


def test_car_handling_is_realistic():
    """Volle Lenkung bei 100 km/h: Rennwagen-Grip (1-2 g) und kein Dreher.
    Schützt davor, dass eine Physik-Änderung das Auto unfahrbar oder unrealistisch macht."""
    import math
    from game.car import DT, Car
    car = Car(0, 0, 0)
    car.vz = 100 / 3.6
    for _ in range(int(3 / DT)):
        vf, _ = car.local_velocity()
        car.update(1.0, 1.0 if vf < 100 / 3.6 else 0.0, 0.0)
    vf, vr = car.local_velocity()
    lateral_g = vf * car.yaw_rate / 9.81
    drift_deg = abs(math.degrees(math.atan2(vr, vf)))
    assert 1.0 < lateral_g < 2.0
    assert drift_deg < 15


def test_records_ignore_old_physics_and_missing_ghost(tmp_path, monkeypatch):
    """Alte Bestzeiten (andere Physik) oder fehlende Geist-Dateien dürfen das Spiel nicht kaputt machen."""
    import json
    import game.ghost as gh
    monkeypatch.setattr(gh, "RECORDS_FILE", tmp_path / "bestzeiten.json")
    gh.RECORDS_FILE.write_text(json.dumps({"map01": {"time": 30.0, "ghost": "gibt/es/nicht.json",
                                                     "physics_version": PHYSICS_VERSION - 1}}))
    assert gh.best_time("map01") is None
    ghost_file = tmp_path / "g.json"
    ghost_file.write_text("{}")
    assert gh.submit_time("map01", 50.0, ghost_file) is True      # alter, schnellerer Rekord zählt nicht
    assert gh.best_time("map01")["time"] == 50.0
    assert gh.submit_time("map01", 55.0, ghost_file) is False
    assert gh.load_records()["alte_rekorde"][0]["time"] == 30.0   # alter Rekord bleibt aufgehoben


def test_car_never_oversteers():
    """Volle Lenkung + Vollbremsung bei 120 km/h: das Heck darf nicht ausbrechen (max. 6° Schräglage)."""
    import math
    from game.car import DT, Car
    car = Car(0, 0, 0)
    car.vz = 120 / 3.6
    worst = 0.0
    for _ in range(int(2.5 / DT)):
        car.update(1.0, 0.0, 1.0)
        vf, vr = car.local_velocity()
        if vf > 5:
            worst = max(worst, -math.degrees(math.atan2(vr, vf)))  # >0 = Heck kommt
    assert worst < 6


def test_drift_mode():
    """Drift (Leertaste): ohne Gegenlenken bricht das Heck aus; loslassen -> Auto fängt sich wieder."""
    import math
    from game.car import DT, Car

    def tail_angle(car):
        vf, vr = car.local_velocity()
        return -math.degrees(math.atan2(vr, max(vf, 0.5)))

    car = Car(0, 0, 0)
    car.vz = 80 / 3.6
    worst = 0.0
    for _ in range(int(2.0 / DT)):
        car.update(1.0, 1.0, 0.0, drift=1.0)
        worst = max(worst, tail_angle(car))
    assert worst > 30                        # Heck kommt

    car = Car(0, 0, 0)
    car.vz = 80 / 3.6
    for _ in range(int(0.4 / DT)):
        car.update(1.0, 1.0, 0.0, drift=1.0)  # kurz andriften
    for _ in range(int(1.5 / DT)):
        car.update(0.0, 0.3, 0.0, drift=0.0)  # loslassen, geradeaus
    assert abs(tail_angle(car)) < 5          # gefangen


def test_ghost_from_old_map_version_is_marked(tmp_path):
    """Wurde eine Map geändert (neue Version), gelten alte Geister als 'alte Version'."""
    sim = Sim(Track.load(TRACKS[0]))
    sim.step(action(throttle=1))
    ghost = Ghost(save_ghost(sim, "player", folder=tmp_path))
    assert ghost.compatible
    ghost.meta["track_version"] = sim.track.version - 1
    assert not ghost.compatible


def test_observation_is_driver_view_only():
    """Regel R5: Die KI sieht dasselbe, egal wo auf der Karte sie steht (keine Koordinaten).
    Gleiche Lage relativ zur Wand -> gleiche Beobachtung, auch wenn die Strecke verschoben ist."""
    import json
    data = json.loads(TRACKS[0].read_text())
    shifted = dict(data, points=[[x + 1000, z - 500] for x, z in data["points"]])
    a = Sim(Track(data, TRACKS[0])).reset()
    b = Sim(Track(shifted, TRACKS[0])).reset()
    assert a.shape == (21,)
    assert np.allclose(a, b, atol=1e-5)
