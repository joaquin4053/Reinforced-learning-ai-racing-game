"""Kamera-Modi: Verfolger, Cockpit, Vogelperspektive und automatische Kinokamera.

Die Kinokamera schneidet alle paar Sekunden zwischen verschiedenen Einstellungen
(wie bei einer TV-Übertragung). Deterministisch über `seed`, damit Videos reproduzierbar sind.
"""
import math
import random

from ursina import Vec3, camera, lerp

MODES = ["chase", "cinematic", "cockpit", "top"]


def _smooth(dt, speed):
    return 1 - math.exp(-speed * dt)


def _look_at(target):
    """Wie camera.look_at, aber ohne Roll (Ursinas look_at kippt die Kamera mit der Zeit)."""
    d = target - camera.position
    camera.rotation = Vec3(math.degrees(math.atan2(-d.y, math.hypot(d.x, d.z))),
                           math.degrees(math.atan2(d.x, d.z)), 0)


class CameraRig:
    def __init__(self, track, mode="chase", seed=0):
        self.track = track
        self.mode = mode
        self.rng = random.Random(seed)
        self.shot = None
        self.shot_time = 0.0
        self.shot_len = 0.0
        self.tv_pos = None
        self.orbit = 0.0
        self.first = True
        self.last_shot = None

    def next_mode(self):
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        self.first = True
        self.shot = None
        return self.mode

    def update(self, x, z, heading, speed, progress, dt):
        fwd = Vec3(math.sin(heading), 0, math.cos(heading))
        right = Vec3(math.cos(heading), 0, -math.sin(heading))
        car = Vec3(x, 0, z)
        getattr(self, "_" + self.mode)(car, fwd, right, speed, progress, dt)
        self.first = False

    # ---------- einfache Modi ----------
    def _follow(self, pos, look, dt, stiff=7.0, fov=None):
        if self.first:
            camera.position = pos
        else:
            camera.position = lerp(camera.position, pos, _smooth(dt, stiff))
        _look_at(look)
        if fov is not None:
            camera.fov = lerp(camera.fov, fov, _smooth(dt, 3.0)) if not self.first else fov

    def _chase(self, car, fwd, right, speed, progress, dt):
        fov = 70 + min(abs(speed), 50) * 0.35  # bei Speed etwas mehr Sichtfeld
        self._follow(car - fwd * 10 + Vec3(0, 4, 0), car + fwd * 6 + Vec3(0, 1, 0), dt, fov=fov)

    def _cockpit(self, car, fwd, right, speed, progress, dt):
        camera.position = car + fwd * 0.3 + Vec3(0, 1.35, 0)
        _look_at(camera.position + fwd * 10 + Vec3(0, -0.6, 0))
        camera.fov = 85

    def _top(self, car, fwd, right, speed, progress, dt):
        self._follow(car + Vec3(0, 70, -0.01) - fwd * 25, car + fwd * 10, dt, stiff=4, fov=60)

    # ---------- Kinokamera ----------
    SHOTS = {  # Name: (min, max) Dauer in Sekunden
        "low_chase": (3.5, 5.0),
        "drone": (4.0, 6.0),
        "tv": (3.0, 5.0),
        "front": (2.0, 3.0),
        "heli": (3.5, 5.0),
    }

    def _pick_shot(self, car, fwd, progress):
        options = [s for s in self.SHOTS if s != self.last_shot]
        shot = self.rng.choice(options) if self.last_shot else "low_chase"  # Start immer von hinten
        self.shot, self.last_shot = shot, shot
        self.shot_time = 0.0
        self.shot_len = self.rng.uniform(*self.SHOTS[shot])
        self.first = True
        self.orbit = self.rng.uniform(0, 360)
        self.orbit_dir = self.rng.choice((-1, 1))
        if shot == "tv":
            # Kamera am Streckenrand ~70 m voraus aufstellen
            p, t = self.track.point_at(progress + 70)
            side = self.rng.choice((-1, 1))
            n = Vec3(t[1], 0, -t[0]) * side
            self.tv_pos = Vec3(p[0], 0, p[1]) + n * (self.track.width / 2 + 8) + Vec3(0, 5, 0)

    def _cinematic(self, car, fwd, right, speed, progress, dt):
        if self.shot is None or self.shot_time > self.shot_len:
            self._pick_shot(car, fwd, progress)
        self.shot_time += dt
        up = Vec3(0, 1, 0)
        s = self.shot
        if s == "low_chase":
            self._follow(car - fwd * 6.5 + right * 1.5 + up * 1.1, car + fwd * 8 + up * 0.8, dt, stiff=10, fov=80)
        elif s == "drone":
            self.orbit += self.orbit_dir * 25 * dt
            a = math.radians(self.orbit)
            pos = car + Vec3(math.sin(a) * 16, 9, math.cos(a) * 16)
            self._follow(pos, car + up, dt, stiff=5, fov=55)
        elif s == "tv":
            camera.position = self.tv_pos
            _look_at(car + up)
            dist = (car - self.tv_pos).length()
            camera.fov = max(12, min(65, math.degrees(2 * math.atan(9 / max(dist, 1)))))  # Tele-Zoom
            if dist > 45 and self.shot_time > 1.5 and (car - self.tv_pos).dot(fwd) > 0:
                self.shot_time = self.shot_len + 1  # Auto ist vorbei -> Schnitt
        elif s == "front":
            self._follow(car + fwd * 9 + right * 1.0 + up * 1.8, car + up * 0.8, dt, stiff=14, fov=60)
        elif s == "heli":
            self._follow(car - fwd * 22 + right * 8 + up * 20, car + fwd * 12, dt, stiff=3, fov=50)
