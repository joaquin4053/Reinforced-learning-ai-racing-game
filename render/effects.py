"""Fahreffekte (nur Anzeige): Reifenspuren und Rauch beim Rutschen.

Rutscht das Auto deutlich (Winkel zwischen Nase und Fahrtrichtung über SLIP_DEG), legen die Hinterräder
dunkle Spuren auf die Fahrbahn und es steigt Rauch auf. Feste Anzahl Objekte, die wiederverwendet
werden (Ringpuffer) -> kein Ruckeln, auch bei langen Drifts.
"""
import math

from ursina import Entity, Vec3, color
from ursina.shaders import unlit_shader

SLIP_DEG = 9.0          # ab so viel Rutschwinkel entstehen Spuren und Rauch
MARK_COUNT = 600        # so viele Spur-Stücke bleiben liegen
SMOKE_COUNT = 60
MARK_EVERY = 0.6        # m Abstand zwischen zwei Spur-Stücken
REAR_OFFSET = (-1.1, 1.1)   # Hinterräder seitlich (m)
REAR_BACK = 1.25            # Hinterachse hinter dem Schwerpunkt (m)


class DriftEffects:
    def __init__(self):
        self.marks = [Entity(model="quad", color=color.rgba(0.05, 0.05, 0.06, 0.55), shader=unlit_shader,
                             rotation_x=90, scale=(0.5, MARK_EVERY + 0.15), enabled=False)
                      for _ in range(MARK_COUNT)]
        self.smoke = [Entity(model="sphere", color=color.rgba(0.85, 0.86, 0.9, 0.0), shader=unlit_shader,
                             enabled=False) for _ in range(SMOKE_COUNT)]
        self.smoke_life = [0.0] * SMOKE_COUNT
        self.mi = self.si = 0
        self.last = {}                 # letzte Spur-Position je Rad

    def reset(self):
        for m in self.marks:
            m.enabled = False
        for s in self.smoke:
            s.enabled = False
        self.last.clear()

    def update(self, car, dt):
        vf, vr = car.local_velocity()
        speed = math.hypot(vf, vr)
        slip = math.degrees(math.atan2(abs(vr), max(abs(vf), 0.1))) if speed > 6 else 0.0
        active = car.grounded and slip > SLIP_DEG
        fx, fz = car.forward
        rx, rz = car.right
        for side in REAR_OFFSET:
            wx = car.x - fx * REAR_BACK + rx * side
            wz = car.z - fz * REAR_BACK + rz * side
            if not active:
                self.last.pop(side, None)
                continue
            prev = self.last.get(side)
            if prev is None or math.hypot(wx - prev[0], wz - prev[1]) >= MARK_EVERY:
                if prev is not None:
                    self._mark((wx + prev[0]) / 2, car.y + 0.03, (wz + prev[1]) / 2,
                               math.atan2(wx - prev[0], wz - prev[1]))
                self.last[side] = (wx, wz)
                self._puff(wx, car.y + 0.4, wz, min(1.0, slip / 35))
        for i, s in enumerate(self.smoke):            # Rauch steigt auf, wird grösser und blasser
            if not s.enabled:
                continue
            self.smoke_life[i] -= dt
            life = self.smoke_life[i]
            if life <= 0:
                s.enabled = False
                continue
            s.y += 1.2 * dt
            s.scale *= 1 + 1.4 * dt
            s.color = color.rgba(0.85, 0.86, 0.9, 0.45 * life)

    def _mark(self, x, y, z, angle):
        m = self.marks[self.mi]
        self.mi = (self.mi + 1) % MARK_COUNT
        m.enabled = True
        m.position = Vec3(x, y, z)
        m.rotation = Vec3(90, math.degrees(angle), 0)

    def _puff(self, x, y, z, strength):
        s = self.smoke[self.si]
        self.smoke_life[self.si] = 0.6 + 0.6 * strength
        self.si = (self.si + 1) % SMOKE_COUNT
        s.enabled = True
        s.position = Vec3(x, y, z)
        s.scale = 0.6 + 0.6 * strength


STREAK_COUNT = 70
STREAK_FROM = 40.0       # m/s (~150 km/h): ab hier erscheinen Geschwindigkeitsstreifen


class SpeedStreaks:
    """Feine helle Striche in der Luft um die Kamera, an denen man vorbeirast (nur Anzeige, Tempogefühl).

    Fester Zufall (Seed) -> immer gleich. Striche, die hinter die Kamera geraten, springen wieder nach vorne.
    """

    def __init__(self, seed=7):
        import random
        self.rng = random.Random(seed)
        self.streaks = [Entity(model="cube", color=color.rgba(1, 1, 1, 0.0), shader=unlit_shader, enabled=False)
                        for _ in range(STREAK_COUNT)]
        self.offsets = [self._new_offset(spread_forward=True) for _ in self.streaks]

    def _new_offset(self, spread_forward=False):
        r = self.rng
        side = r.choice((-1, 1)) * r.uniform(3.0, 14.0)          # nie direkt vor der Nase
        up = r.uniform(-1.0, 7.0)
        ahead = r.uniform(5.0, 90.0) if spread_forward else r.uniform(60.0, 90.0)
        return [side, up, ahead]

    def update(self, car, cam_pos, dt):
        speed = car.speed
        k = max(0.0, min(1.0, (speed - STREAK_FROM) / 50.0))      # 0 bei 150 km/h, 1 ab ~330 km/h
        if speed > 1:
            dx, dz = car.vx / speed, car.vz / speed
        else:
            dx, dz = car.forward
        rx, rz = dz, -dx
        visible = int(STREAK_COUNT * k)
        for i, (e, off) in enumerate(zip(self.streaks, self.offsets)):
            off[2] -= speed * dt                                     # Strich bleibt in der Welt stehen
            if off[2] < -3.0:
                self.offsets[i] = off = self._new_offset()
            if i >= visible:
                e.enabled = False
                continue
            e.enabled = True
            length = 1.0 + speed * 0.06
            e.position = Vec3(cam_pos.x + dx * off[2] + rx * off[0], cam_pos.y + off[1] - 1.5,
                              cam_pos.z + dz * off[2] + rz * off[0])
            e.rotation = Vec3(0, math.degrees(math.atan2(dx, dz)), 0)
            e.scale = Vec3(0.04, 0.04, length)
            e.color = color.rgba(1, 1, 1, 0.35 * k)
