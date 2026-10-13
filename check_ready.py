"""Prüfprogramm: Ist die Simulation bereit fürs KI-Training?

    .venv/bin/python check_ready.py

Läuft ohne Fenster. Gibt für jede Prüfung OK oder FEHLER mit Erklärung aus.
Am Ende: Zusammenfassung. Rückgabewert 0 = alles OK, 1 = mindestens ein Fehler.
"""
import math
import random
import sys
import time

from game.car import DT
from game.sim import (PHYSICS_VERSION, RULES_VERSION, SENSOR_VERSION, Rules, Sim, action,
                      format_time)
from game.track import Track
from tools.autopilot_run import autopilot

TRACK_FILE = "tracks/map01.json"
# Für Prüfungen, die lange fahren müssen: Fahrt endet nie von selbst (kein "stecken geblieben" usw.)
ENDLESS = Rules(stuck_seconds=1e9, wrong_way_meters=1e9, time_limit=1e9)

CHECKS = []


def check(title):
    """Markiert eine Funktion als Prüfung. Sie gibt (ok, erklärung) zurück."""
    def register(func):
        CHECKS.append((title, func))
        return func
    return register


# ---------- Hilfsfunktionen ----------
def random_actions(n, seed):
    """Feste "zufällige" Eingabefolge – gleicher Seed = gleiche Folge."""
    rng = random.Random(seed)
    return [[rng.uniform(-1, 1), rng.random(), rng.random() * 0.3, float(rng.random() < 0.1)] for _ in range(n)]


def full_state(sim):
    """Alles, was den Zustand der Simulation ausmacht (für exakte Vergleiche)."""
    c = sim.car
    return (c.x, c.z, c.heading, c.vx, c.vz, c.yaw_rate, c.steer, c.drift, c.accel,
            sim.frame, sim.progress, sim.wall_hits, tuple(sim.splits))


def drive_lap(sim):
    """Testfahrer (Autopilot) fährt eine Runde. Gibt den Fortschritt nach jedem Schritt zurück."""
    progress = [sim.progress]
    while not sim.done:
        sim.step(autopilot(sim, 60))
        progress.append(sim.progress)
    return progress


def place_car(sim, dist, angle=0.0, speed=20.0):
    """Auto an eine Stelle der Strecke setzen (dist = Meter ab Start), angle = Abweichung von der Fahrtrichtung."""
    (x, z), (tx, tz) = sim.track.point_at(dist)
    car = sim.car
    car.x, car.z = float(x), float(z)
    car.heading = math.atan2(tx, tz) + angle
    fx, fz = car.forward
    car.vx, car.vz = fx * speed, fz * speed
    car.yaw_rate = 0.0


def inside_track(track, x, z):
    _, _, _, lateral = track.progress_at(x, z)
    return abs(lateral) < track.width / 2


# ---------- Prüfungen ----------
@check("Simulation startet ohne Grafik")
def check_no_graphics():
    sim = Sim(Track.load(TRACK_FILE))
    sim.reset()
    sim.step(action(throttle=1))
    loaded = [m for m in ("ursina", "panda3d") if m in sys.modules]
    if loaded:
        return False, f"Grafik-Bibliotheken wurden geladen: {', '.join(loaded)}"
    return True, "Simulation läuft, ohne dass Ursina/Panda3D geladen wurden."


@check("Geschwindigkeit")
def check_speed():
    sim = Sim(Track.load(TRACK_FILE), rules=ENDLESS, record=False)
    actions = random_actions(10_000, seed=1)
    start = time.perf_counter()
    for a in actions:
        sim.step(a)
        sim.observation()
    per_second = len(actions) / (time.perf_counter() - start)
    ok = per_second >= 3000
    nice = f"{per_second:,.0f}".replace(",", "'")
    return ok, f"{nice} Schritte pro Sekunde (mit Sensoren), Ziel: mehrere tausend."


@check("Determinismus")
def check_determinism():
    track = Track.load(TRACK_FILE)
    actions = random_actions(2000, seed=42)
    ends = []
    for _ in range(2):
        sim = Sim(track, rules=ENDLESS)
        for a in actions:
            sim.step(a)
        ends.append((full_state(sim), sim.history))
    if ends[0] != ends[1]:
        return False, "Gleiche Eingaben haben zu unterschiedlichen Fahrten geführt."
    x, z = ends[0][0][:2]
    return True, f"2000 zufällige Schritte (Seed 42) zweimal gefahren: Endzustand exakt gleich (Auto bei {x:.3f}, {z:.3f})."


@check("reset() stellt das Auto exakt an den Start")
def check_reset():
    track = Track.load(TRACK_FILE)
    sim = Sim(track, rules=ENDLESS)
    for a in random_actions(300, seed=3):
        sim.step(a)
    sim.reset()
    c = sim.car
    problems = []
    if (c.x, c.z) != track.start_pos:
        problems.append(f"Position {c.x:.3f},{c.z:.3f} statt {track.start_pos}")
    if c.heading != track.start_heading:
        problems.append("Richtung stimmt nicht")
    if c.speed != 0 or c.yaw_rate != 0 or c.steer != 0:
        problems.append(f"Auto steht nicht still (Tempo {c.speed:.3f} m/s)")
    if sim.frame != 0 or sim.time != 0 or sim.splits or sim.done:
        problems.append("Zeit/Checkpoints nicht zurückgesetzt")
    if problems:
        return False, "; ".join(problems)
    return True, "Nach 300 Schritten Fahrt: reset() setzt Position, Richtung, Tempo 0, Zeit und Checkpoints exakt zurück."


@check("Fortschritt entlang der Strecke")
def check_progress():
    track = Track.load(TRACK_FILE)
    sim = Sim(track)
    start_pct = sim.progress_pct
    progress = drive_lap(sim)
    if sim.result != "finished":
        return False, f"Testfahrer ist nicht ins Ziel gekommen ({sim.end_reason})."
    max_step = 60 * DT + 0.5    # mehr als ~60 m/s * Zeitschritt kann das Auto nicht fahren
    jumps = [b - a for a, b in zip(progress, progress[1:]) if abs(b - a) > max_step]
    if start_pct > 1.0:
        return False, f"Am Start schon {start_pct:.1f} % (sollte ca. 0 % sein)."
    if jumps:
        return False, f"Fortschritt springt {len(jumps)}x, z.B. um {jumps[0]:.1f} m in einem Schritt."
    if abs(progress[-1] - track.length) > 5:
        return False, f"Im Ziel {progress[-1]:.0f} m statt Streckenlänge {track.length:.0f} m."
    return True, (f"Start {start_pct:.1f} %, Ziel {sim.progress_pct:.0f} % ({progress[-1]:.0f} von "
                  f"{track.length:.0f} m), keine Sprünge (größter Schritt {max(abs(b - a) for a, b in zip(progress, progress[1:])):.2f} m).")


@check("R1: nur gültige Runden zählen")
def check_r1():
    track = Track.load(TRACK_FILE)
    # a) normale Runde: alle Checkpoints in Reihenfolge -> gültig
    sim = Sim(track)
    drive_lap(sim)
    if sim.result != "finished" or len(sim.splits) != len(track.checkpoints) + 1:
        return False, "Eine normale Runde mit allen Checkpoints wurde nicht als gültig gewertet."
    # b) direkt zur Ziellinie, ohne Checkpoints
    sim = Sim(track, rules=ENDLESS)
    place_car(sim, track.finish_gate.dist - 5)
    for _ in range(60):
        sim.step(action(throttle=1))
    if sim.result == "finished" or sim.splits:
        return False, "Ziellinie ohne Checkpoints überquert -> wurde trotzdem gewertet!"
    # c) ersten Checkpoint fahren, zweiten überspringen
    sim = Sim(track, rules=ENDLESS)
    while len(sim.splits) < 1:
        sim.step(autopilot(sim, 60))
    place_car(sim, track.checkpoints[2].dist - 5)          # hinter Checkpoint 2 "beamen"
    for _ in range(60):
        sim.step(action(throttle=1))
    place_car(sim, track.finish_gate.dist - 5)
    for _ in range(60):
        sim.step(action(throttle=1))
    if len(sim.splits) != 1 or sim.result == "finished":
        return False, "Ein übersprungener Checkpoint wurde nicht erkannt."
    return True, ("Volle Runde = gültig mit Zeit. Ziel ohne Checkpoints = keine Zeit. "
                  "Checkpoint übersprungen = spätere Checkpoints und Ziel zählen nicht.")


@check("R2: Wandberührung wird erkannt, Fahrbahn kann nicht verlassen werden")
def check_r2():
    track = Track.load(TRACK_FILE)
    spots = [track.length * k / 8 + 10 for k in range(8)]
    for dist in spots:
        for side in (-1, 1):  # nach links und nach rechts in die Wand fahren
            sim = Sim(track, rules=ENDLESS)
            place_car(sim, dist, angle=side * math.radians(70), speed=15)
            hits = 0
            for _ in range(120):
                hits += sim.step(action(throttle=0.6))["wall_hit"]
                if not inside_track(track, sim.car.x, sim.car.z):
                    return False, f"Auto hat bei {dist:.0f} m die Fahrbahn verlassen!"
            if hits == 0:
                return False, f"Bei {dist:.0f} m wurde keine Wandberührung gemeldet."
            if sim.done:
                return False, "Wandberührung hat die Fahrt beendet (laut R2 soll das Auto abprallen)."
    # Langer Test: 20'000 wilde Schritte, das Auto darf nie außerhalb der Fahrbahn sein
    sim = Sim(track, rules=ENDLESS, record=False)
    for a in random_actions(20_000, seed=7):
        sim.step([a[0], 1.0, 0.0, a[3]])
        if not inside_track(track, sim.car.x, sim.car.z):
            return False, f"Nach {sim.frame} wilden Schritten außerhalb der Fahrbahn."
    return True, (f"An {len(spots)} Stellen links und rechts gegen die Wand gefahren: jede Berührung gemeldet, "
                  f"Auto prallt ab, Fahrt geht weiter (R2, geändert mit Zustimmung). 20'000 wilde Schritte: "
                  f"nie außerhalb der Fahrbahn.")


@check("R4: Aktionen werden auf den erlaubten Bereich abgeschnitten")
def check_r4():
    track = Track.load(TRACK_FILE)
    pairs = [
        ([5.0, 9.0, -3.0, 7.0], [1.0, 1.0, 0.0, 1.0]),       # zu groß / zu klein
        ([-8.0, -2.0, 4.0, -1.0], [-1.0, 0.0, 1.0, 0.0]),
        ([float("nan")] * 4, [0.0, 0.0, 0.0, 0.0]),            # ungültig (NaN) zählt als 0
    ]
    for wild, clipped in pairs:
        a, b = Sim(track, rules=ENDLESS), Sim(track, rules=ENDLESS)
        for _ in range(300):
            a.step(wild)
            b.step(clipped)
            if not -1.0 <= a.car.steer <= 1.0:
                return False, f"Lenkung außerhalb des Bereichs: {a.car.steer}"
        if full_state(a) != full_state(b):
            return False, f"Eingabe {wild} verhält sich nicht wie {clipped}."
    return True, "Zu große, zu kleine und ungültige Werte (NaN) fahren exakt wie die abgeschnittenen Werte."


@check("R6: Rundenzeit = Physik-Schritte × Zeitschritt")
def check_r6():
    track = Track.load(TRACK_FILE)
    times = []
    for slow in (False, True):   # einmal normal, einmal mit künstlichen Pausen (echte Uhr darf nichts ändern)
        sim = Sim(track)
        while not sim.done:
            sim.step(autopilot(sim, 60))
            if slow and sim.frame % 300 == 0:
                time.sleep(0.01)
        times.append((sim.finish_time, sim.frame, list(sim.splits)))
    (t, frames, splits), (t2, _, _) = times
    if t != frames * DT:
        return False, f"Zeit {t} ist nicht {frames} × {DT}."
    if t != t2:
        return False, "Rundenzeit hängt von der echten Uhr ab."
    if any(abs(s / DT - round(s / DT)) > 1e-6 for s in splits):
        return False, "Zwischenzeiten sind keine ganzen Physik-Schritte."
    if format_time(41.6499) != "0:41.65":
        return False, f"Anzeige nicht auf 0.01 s: {format_time(41.6499)}"
    return True, (f"Rundenzeit {format_time(t)} = {frames} Schritte × 1/{round(1 / DT)} s, "
                  f"unabhängig von der echten Uhr, Anzeige auf 0.01 s.")


# ---------- Ausführen ----------
def main():
    print(f"Prüfe Simulation  (Physik v{PHYSICS_VERSION}, Regeln v{RULES_VERSION}, Sensoren v{SENSOR_VERSION}, "
          f"Strecke {TRACK_FILE})\n")
    failed = 0
    for title, func in CHECKS:
        try:
            ok, text = func()
        except Exception as e:  # Absturz in einer Prüfung = FEHLER, die anderen laufen weiter
            ok, text = False, f"Absturz: {type(e).__name__}: {e}"
        failed += not ok
        print(f"{'OK    ' if ok else 'FEHLER'}  {title}")
        print(f"        {text}\n")
    if failed:
        print(f"{failed} von {len(CHECKS)} Prüfungen FEHLGESCHLAGEN.")
    else:
        print(f"Alle {len(CHECKS)} Prüfungen OK – die Simulation ist bereit.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
