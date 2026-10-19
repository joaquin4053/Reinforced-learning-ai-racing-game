"""Tastatur + Gamepad -> action-Dict, genau das gleiche Format wie bei der KI.

- Gamepad: linker Stick lenkt stufenlos, rechter Trigger = Gas, linker Trigger = Bremse,
  X oder rechte Schultertaste = Drift.
- Leertaste = Drift (Heck bricht aus, man muss gegenlenken).
- Tastatur: Lenkung wird leicht geglättet (fährt in ~0.22 s auf Vollausschlag),
  damit man auch mit Tasten sanft lenken kann.
Wird pro Physik-Schritt aufgerufen (dt = 1/60 s).
"""
from ursina import held_keys

from game.sim import action

STEER_RATE = 4.5      # Tastatur: so schnell geht die Lenkung zum Ziel (pro Sekunde, 4.5 = in 0.22 s voll)
RETURN_RATE = 10.0    # ...und so schnell zurück in die Mitte
DEADZONE = 0.08       # Gamepad-Stick: kleine Ausschläge ignorieren

RESTART_KEYS = ("r", "t", "backspace", "gamepad y", "gamepad start")


def _approach(value, target, step):
    if value < target:
        return min(value + step, target)
    return max(value - step, target)


class Controls:
    def __init__(self):
        self.steer = 0.0

    def reset(self):
        self.steer = 0.0

    def read(self, dt):
        k = held_keys
        key_steer = (k["d"] or k["right arrow"]) - (k["a"] or k["left arrow"])
        stick = k["gamepad left stick x"]

        if abs(stick) > DEADZONE:  # Gamepad hat Vorrang, stufenlos
            sign = 1 if stick > 0 else -1
            self.steer = sign * (abs(stick) - DEADZONE) / (1 - DEADZONE)
        else:
            back_to_center = key_steer == 0 or (self.steer != 0 and (key_steer > 0) != (self.steer > 0))
            rate = RETURN_RATE if back_to_center else STEER_RATE
            self.steer = _approach(self.steer, key_steer, rate * dt)

        throttle = max(k["w"], k["up arrow"], k["gamepad right trigger"], k["gamepad a"])
        brake = max(k["s"], k["down arrow"], k["gamepad left trigger"], k["gamepad b"])
        drift = max(k["space"], k["gamepad x"], k["gamepad right shoulder"])
        return action(steer=self.steer, throttle=float(throttle), brake=float(brake), drift=float(drift))
