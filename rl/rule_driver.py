"""Einfacher Regel-Fahrer: sieht NUR die Beobachtung der KI (Fahrersicht, R5).

Zeigt, dass die Strecke mit diesen Sensoren lösbar ist, und dient später als Vergleich.
  - Lenken: in die Richtung, in der die Strahlen am meisten Platz zeigen
  - Gas: mäßig, je weniger Platz vorne, desto langsamer
"""
import numpy as np

from game import sensors

_ANGLES = sensors.RAY_ANGLES_DEG
_FRONT = sensors.RAY_COUNT // 2          # Strahl geradeaus
_TEMPO = [row[0] for row in sensors.LAYOUT].index("tempo")   # Index "tempo" (siehe Tabelle in CLAUDE.md)


def rule_action(obs):
    rays_m = obs[:sensors.RAY_COUNT] * sensors.RAY_RANGE
    weights = rays_m ** 4                                     # lange Strahlen zählen viel mehr
    total = float(np.sum(weights))
    target_deg = float(np.sum(_ANGLES * weights) / total) if total > 0 else 0.0   # alles 0: z.B. über einer Lücke
    steer = np.clip(target_deg / 25.0, -1.0, 1.0)

    speed = obs[_TEMPO] * sensors.SPEED_NORM                 # m/s
    front = rays_m[_FRONT]
    target_speed = min(28.0, 6.0 + 0.3 * front)              # mäßig: höchstens ~100 km/h
    throttle = 0.7 if speed < target_speed else 0.0
    brake = 0.6 if speed > target_speed + 4 else 0.0
    return np.array([steer, throttle, brake, 0.0], dtype=np.float32)
