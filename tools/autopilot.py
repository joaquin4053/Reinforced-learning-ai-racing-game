"""Testfahrer, der die Strecke KENNT (schummelt) – nur für Tests und Vorschau, keine KI!

    from tools.autopilot import autopilot
    sim.step(autopilot(sim, max_speed=40))

Lenkt auf einen Punkt voraus auf der Mittellinie und bremst vor engen Kurven.
"""
import math

from game.sim import action


def autopilot(sim, max_speed=60.0):
    track, car = sim.track, sim.car
    vf, _ = car.local_velocity()
    look = 8 + 0.35 * max(vf, 0)
    (tx, tz), _ = track.point_at(sim.progress + look)
    ang = math.atan2(tx - car.x, tz - car.z) - car.heading
    ang = (ang + math.pi) % (2 * math.pi) - math.pi
    steer = ang * 3.5
    safe = max_speed
    _, t0 = track.point_at(sim.progress)
    for d in range(10, 130, 10):
        _, t1 = track.point_at(sim.progress + d)
        turn = abs(math.atan2(t0[0] * t1[1] - t0[1] * t1[0], t0[0] * t1[0] + t0[1] * t1[1]))
        if turn > 0.05:
            corner = math.sqrt(12.0 * d / turn)
            safe = min(safe, math.sqrt(corner ** 2 + 2 * 10.0 * max(d - 15, 0)))
    throttle = 1.0 if vf < safe else 0.0
    brake = 1.0 if vf > safe + 2 else 0.0
    if not car.grounded:
        steer = 0.0
    return action(steer=max(-1, min(1, steer)), throttle=throttle, brake=brake)
