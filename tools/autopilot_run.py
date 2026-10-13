"""Testfahrt ohne Grafik: ein einfacher Autopilot folgt der Mittellinie und speichert einen Geist.

    python tools/autopilot_run.py [tracks/map01.json] [--speed 30] [--wobble 0.0] [--attempt 1]

Nützlich, um Videos, Kameras und Geister zu testen, ohne selbst zu fahren.
--wobble > 0 macht den Fahrer unsauber (zum Testen von Crashs). Zufall nur mit --seed.
"""
import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from game.ghost import save_ghost  # noqa: E402
from game.sim import Sim, action  # noqa: E402
from game.track import Track  # noqa: E402


def autopilot(sim, target_speed, wobble=0.0, rng=None):
    """Lenkt auf einen Punkt voraus auf der Mittellinie und bremst vor engen Kurven.

    Kennt die Strecke (schummelt also) – nur zum Testen, keine KI!
    """
    track, car = sim.track, sim.car
    vf, _ = car.local_velocity()
    look = 10 + 0.4 * max(vf, 0)                    # bei Tempo weiter vorausschauen
    (tx, tz), _ = track.point_at(sim.progress + look)
    ang = math.atan2(tx - car.x, tz - car.z) - car.heading
    ang = (ang + math.pi) % (2 * math.pi) - math.pi
    steer = ang * 4 + (rng.gauss(0, wobble) if wobble else 0.0)

    # Kurven voraus: wie schnell darf ich dort sein? (v² = Querbeschleunigung · Radius)
    safe = target_speed
    _, t0 = track.point_at(sim.progress)
    for d in range(10, 120, 10):
        _, t1 = track.point_at(sim.progress + d)
        turn = abs(math.atan2(t0[0] * t1[1] - t0[1] * t1[0], t0[0] * t1[0] + t0[1] * t1[1]))
        if turn > 0.05:
            radius = d / turn
            corner_speed = math.sqrt(11.0 * radius)
            safe = min(safe, math.sqrt(corner_speed ** 2 + 2 * 9.0 * max(d - 15, 0)))  # Bremsweg
    throttle = 1.0 if vf < safe else 0.0
    brake = 1.0 if vf > safe + 2 else 0.0
    return action(steer=max(-1, min(1, steer)), throttle=throttle, brake=brake)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("track", nargs="?", default="tracks/map01.json")
    ap.add_argument("--speed", type=float, default=60.0, help="Höchstgeschwindigkeit in m/s")
    ap.add_argument("--wobble", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attempt", type=int, default=0, help="Versuchszahl im Geist")
    args = ap.parse_args()

    sim = Sim(Track.load(args.track))
    rng = random.Random(args.seed)
    while not sim.done:
        sim.step(autopilot(sim, args.speed, args.wobble, rng))
    path = save_ghost(sim, "ai", attempt=args.attempt, driver="Autopilot")
    print(f"{sim.result} {sim.end_reason or ''} zeit={sim.time:.2f}s fortschritt={sim.progress_pct:.1f}% "
          f"splits={[round(s, 2) for s in sim.splits]} -> {path}")
