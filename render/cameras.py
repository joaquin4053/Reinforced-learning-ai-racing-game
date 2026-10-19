"""Kameras (Spiel v2, 3D): Verfolger, Cockpit, Vogel. Taste C wechselt.

Verfolger (Notizen: docs/beobachtung_polytrack.md):
- hinter dem Auto, leicht erhöht, folgt weich auch in der Höhe,
- blickt zwischen Nase und Fahrtrichtung -> beim Driften sieht man das Auto schräg,
- neigt sich mit der Fahrbahn (Steilkurven).
"""
import math

from ursina import Vec3, camera

MODES = ("chase", "cockpit", "top")
MOVE_BLEND = 0.55      # 0 = Kamera schaut wie die Nase, 1 = wie die Fahrtrichtung
ROLL_FOLLOW = 0.7      # so viel der Querneigung macht die Kamera mit
SHAKE = 0.05           # m Vibrieren der Kamera bei Spitzentempo (nur Anzeige)


def _smooth(dt, speed):
    return 1.0 - math.exp(-speed * dt)


def _angle_lerp(a, b, t):
    return a + ((b - a + math.pi) % (2 * math.pi) - math.pi) * t


class CameraRig:
    def __init__(self, track, mode="chase"):
        self.track = track
        self.mode = mode
        self.first = True
        self.pos = Vec3(0, 0, 0)
        self.look = Vec3(0, 0, 0)
        self.yaw = 0.0
        self.roll = 0.0
        self.t = 0.0

    def next_mode(self):
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        self.first = True

    def update(self, x, y, z, heading, speed, dt, roll=0.0, move_heading=None):
        car = Vec3(x, y, z)
        target_yaw = heading
        if move_heading is not None and speed > 8:
            target_yaw = _angle_lerp(heading, move_heading, MOVE_BLEND)
        if self.first:
            self.yaw, self.roll = target_yaw, roll
        self.yaw = _angle_lerp(self.yaw, target_yaw, _smooth(dt, 6.0))
        fx, fz = math.sin(self.yaw), math.cos(self.yaw)
        tilt = 0.0
        if self.mode == "chase":
            v = min(max(speed, 0), 110)
            back = 6.2 + v * 0.012                                  # tief und nah: Boden rast vorbei
            want_pos = car + Vec3(-fx * back, 2.1, -fz * back)
            want_look = car + Vec3(fx * 10, 0.9, fz * 10)
            stiff, fov = 10.0, 62 + v * 0.30                        # Blickwinkel weitet sich stark mit dem Tempo
            tilt = roll * ROLL_FOLLOW
        elif self.mode == "cockpit":
            hx, hz = math.sin(heading), math.cos(heading)
            want_pos = car + Vec3(hx * 0.3, 1.35, hz * 0.3)
            want_look = car + Vec3(hx * 20, 1.0, hz * 20)
            stiff, fov = 40.0, 85
            tilt = roll
        else:
            want_pos = car + Vec3(-fx * 4, 80, -fz * 4)
            want_look = car + Vec3(fx * 6, 0, fz * 6)
            stiff, fov = 5.0, 60
        if self.first:
            self.pos, self.look = want_pos, want_look
            self.first = False
        self.pos += (want_pos - self.pos) * _smooth(dt, stiff)
        self.look += (want_look - self.look) * _smooth(dt, stiff * 1.6)
        self.roll += (tilt - self.roll) * _smooth(dt, 5.0)
        self.t += dt
        shake = max(0.0, speed - 40) / 70 * SHAKE if self.mode == "chase" else 0.0   # leichtes Vibrieren
        wobble = Vec3(math.sin(self.t * 37.0), math.sin(self.t * 43.0 + 1.3), 0) * shake
        camera.position = self.pos + wobble
        camera.look_at(self.look)
        camera.rotation_z = math.degrees(self.roll)     # gleiche Richtung wie das Auto (geprüft im Bild)
        camera.fov += (fov - camera.fov) * _smooth(dt, 3.0)


class FreeCamera:
    """Frei fliegende Kamera (viewer.py): Maus ziehen = umsehen, W/A/S/D = fliegen, Q/E = runter/hoch,
    Shift = schneller, Mausrad = Fluggeschwindigkeit."""

    def __init__(self):
        self.yaw = self.pitch = 0.0
        self.speed = 40.0          # m/s

    def start(self):
        """Dort weitermachen, wo die Kamera gerade ist."""
        f = camera.forward
        self.yaw = math.atan2(f.x, f.z)
        self.pitch = math.asin(max(-1.0, min(1.0, f.y)))
        camera.rotation_z = 0

    def look(self, dx, dy):
        self.yaw += dx * 3.0
        self.pitch = max(-1.45, min(1.45, self.pitch + dy * 3.0))

    def zoom(self, steps):
        self.speed = max(5.0, min(400.0, self.speed * (1.25 ** steps)))

    def update(self, dt, held):
        cp = math.cos(self.pitch)
        fwd = Vec3(math.sin(self.yaw) * cp, math.sin(self.pitch), math.cos(self.yaw) * cp)
        right = Vec3(math.cos(self.yaw), 0, -math.sin(self.yaw))
        move = fwd * (held["w"] - held["s"]) + right * (held["d"] - held["a"]) + Vec3(0, 1, 0) * (held["e"] - held["q"])
        k = self.speed * (4.0 if held["shift"] else 1.0) * dt
        camera.position += move * k
        camera.look_at(camera.position + fwd)
        camera.rotation_z = 0
        camera.fov += (70 - camera.fov) * _smooth(dt, 3.0)
