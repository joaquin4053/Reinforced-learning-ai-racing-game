"""Fahrphysik in 3D, Arcade-Stil: Reifenmodell in der Ebene + Höhe folgt der Fahrbahn + Flugphasen.

Idee:
- Am Boden: Vorder- und Hinterreifen erzeugen Seitenkräfte ("Einspurmodell"). Bis zum Limit greifen sie
  fest, darüber rutscht das Auto. Die Höhe folgt der Fahrbahn, Steigungen bremsen bzw. beschleunigen.
- Abheben: Fällt die Fahrbahn schneller weg, als die Schwerkraft das Auto nach unten zieht (Rampenkante,
  schnelle Kuppe, Ende der Strecke), fliegt das Auto. In der Luft wirken nur Schwerkraft und Luftwiderstand.
- Landen: Das senkrechte Tempo relativ zur Fahrbahn geht verloren, eine harte Landung kostet etwas Tempo.

Keine Grafik, kein Zufall: gleiche Eingaben -> exakt gleiche Bewegung. Fester Zeitschritt DT (1/60 s).
Koordinaten: x, z waagrecht, y = Höhe. Heading h: Vorwärtsrichtung = (sin h, cos h), h=0 -> +z.
Im Auto gedacht: vf = vorwärts, vr = nach rechts, yaw_rate > 0 = dreht nach rechts.

Alle Zahlenwerte stehen in config.toml (Physik-Version erhöhen bei Änderungen, Regel R8).
"""
import math

from .config import CONFIG

DT = 1.0 / CONFIG["simulation"]["physik_hz"]
SUBSTEPS = CONFIG["simulation"]["unterschritte"]
GRAVITY = CONFIG["simulation"]["gravitation"]
LIFT = 0.05         # m: so weit muss sich das Auto vom Boden lösen, bis es als "in der Luft" gilt
                    #    (kleinere Hüpfer über Rasterkanten fängt der Boden sofort wieder auf)
SETTLE_TIME = 0.4   # s nach einer Landung federt das Auto nach: ...
SETTLE_LIFT = 0.4   # m ... so lange braucht es mehr Abstand, um wieder abzuheben (kein Nachhüpfen wie ein Ball)
VY_SMOOTH = 0.03    # s: Glättung des Tempos nach oben am Boden (sonst kostet die letzte Rasterzelle an einer
                    #    Rampenkante den Schwung)


class CarParams:
    """Alle Fahrzeug-Werte aus config.toml."""

    SECTIONS = ("auto", "antrieb", "reifen", "lenkung", "drift", "luft", "kollision")

    def __init__(self):
        for section in self.SECTIONS:
            for name, value in CONFIG[section].items():
                setattr(self, name, value)

    @property
    def wheelbase(self):
        return self.cg_to_front + self.cg_to_rear


def clip(value, low, high):
    """Wert in den erlaubten Bereich schneiden (Regel R4). Ungültige Werte (NaN) zählen als 0."""
    value = float(value)
    if value != value:
        return 0.0
    return max(low, min(high, value))


def _tire_force(slip, stiffness, max_force, slide_grip):
    """Seitenkraft eines Reifens abhängig vom Schräglaufwinkel (vereinfachte "Magic Formula")."""
    if max_force <= 0:
        return 0.0
    shape = 2.0 / math.pi * math.acos(-slide_grip) if slide_grip < 1 else 1.0
    b = stiffness / (max_force * shape)
    return -max_force * math.sin(shape * math.atan(b * slip))


class Car:
    def __init__(self, x, y, z, heading, params=None):
        self.p = params or CarParams()
        self._max_slope = math.tan(math.radians(self.p.max_steigung))
        self.x, self.y, self.z, self.heading = x, y, z, heading
        self.vx = self.vz = 0.0      # waagrechte Geschwindigkeit (Welt)
        self.vy = 0.0                # senkrechte Geschwindigkeit
        self.yaw_rate = 0.0
        self.steer = 0.0             # Lenkradstellung -1..1
        self.drift = 0.0             # Drift-Modus 0..1
        self.accel = 0.0             # geglättete Längsbeschleunigung (Gewichtsverlagerung)
        self.pitch = 0.0             # rad, > 0 = Nase oben
        self.roll = 0.0              # rad, > 0 = rechte Seite tiefer
        self.grounded = True
        self.airtime = 0.0           # s seit dem Abheben
        self.pitch_rate = 0.0        # rad/s Überschlagen in der Luft (nach vorne/hinten)
        self.roll_rate = 0.0         # rad/s ... und zur Seite
        self.hit = False             # in diesem Schritt gegen eine Blockseite geprallt?
        self.landed = 0.0            # Aufprall-Tempo der letzten Landung (m/s), nur im Landeschritt > 0
        self.settle = 0.0            # s Federung nach der Landung (siehe SETTLE_TIME)

    @property
    def forward(self):
        return math.sin(self.heading), math.cos(self.heading)

    @property
    def right(self):
        return math.cos(self.heading), -math.sin(self.heading)

    @property
    def speed(self):
        return math.hypot(self.vx, self.vz)

    def local_velocity(self):
        """(vorwärts, seitwärts) waagrechte Geschwindigkeit in m/s, relativ zum Auto."""
        fx, fz = self.forward
        rx, rz = self.right
        return self.vx * fx + self.vz * fz, self.vx * rx + self.vz * rz

    # ---------- ein Physik-Schritt ----------
    def update(self, steer, throttle, brake, drift, surface, dt=DT):
        """steer -1..1, throttle 0..1, brake 0..1, drift 0..1. surface.ground(x, z) -> Höhe oder None."""
        steer, throttle = clip(steer, -1.0, 1.0), clip(throttle, 0.0, 1.0)
        brake, drift = clip(brake, 0.0, 1.0), clip(drift, 0.0, 1.0)
        step = self.p.steer_rate * dt
        self.steer += max(-step, min(step, steer - self.steer))
        step = self.p.drift_rate * dt
        self.drift += max(-step, min(step, drift - self.drift))
        self.hit, self.landed = False, 0.0
        self.settle = max(0.0, self.settle - dt)
        was_grounded, pitch0, roll0 = self.grounded, self.pitch, self.roll
        if self.grounded:
            self._tilt(surface)
            self.pitch_rate = (self.pitch - pitch0) / dt      # wie schnell sich die Fahrbahn gerade neigt
            self.roll_rate = (self.roll - roll0) / dt
        sub = dt / SUBSTEPS
        for _ in range(SUBSTEPS):
            if self.grounded:
                self._drive(throttle, brake, sub)
            else:
                self._fly(sub)
            self._move(surface, sub)
        if not self.grounded:
            if was_grounded:                                     # beim Abheben: Drehung der Fahrbahn mitnehmen,
                vf, vr = self.local_velocity()                   # seitliches Rutschen über die Kante kippt zur Seite
                self.roll_rate += vr * 0.08
            self.airtime += dt
            self.pitch += self.pitch_rate * dt                   # in der Luft weiterdrehen (Überschlag)
            self.roll += self.roll_rate * dt
            spin = math.exp(-self.p.spin_damping * dt)
            self.pitch_rate *= spin
            self.roll_rate *= spin
            k = 1.0 - math.exp(-self.p.level_rate * dt)          # langsam waagrecht werden
            self.pitch -= self.pitch * k
            self.roll -= self.roll * k

    def _tilt(self, surface):
        """Neigung aus der Fahrbahnhöhe unter den vier Rad-Positionen."""
        p, fx, fz = self.p, *self.forward
        rx, rz = self.right
        h0 = self.y
        def h(dx, dz):
            g = surface.ground(self.x + dx, self.z + dz)
            return h0 if g is None or abs(g - h0) > p.step_height else g
        hf, hb = h(fx * p.cg_to_front, fz * p.cg_to_front), h(-fx * p.cg_to_rear, -fz * p.cg_to_rear)
        w = p.track_width / 2
        hl, hr = h(-rx * w, -rz * w), h(rx * w, rz * w)
        self.pitch = math.atan2(hf - hb, p.wheelbase)
        self.roll = math.atan2(hl - hr, p.track_width)

    def _drive(self, throttle, brake, dt):
        """Reifen, Motor, Bremse am Boden (wie das bewährte Modell aus Spiel v1) + Hangabtrieb."""
        p = self.p
        vx, vy = self.local_velocity()
        r = self.yaw_rate
        L, a, b = p.wheelbase, p.cg_to_front, p.cg_to_rear

        max_angle = p.max_steer / (1.0 + (vx / p.steer_speed_falloff) ** 2)
        if self.steer * r < 0:
            max_angle += (p.drift_max_steer - max_angle) * self.drift
        delta = self.steer * max_angle

        drive = throttle * min(p.max_drive_force, p.engine_power / max(vx, 1.0))
        braking = 0.0
        if brake > 0:
            if vx > 0.5:
                braking = brake * p.brake_force
            elif vx > -p.max_reverse:
                drive -= brake * p.reverse_force
        resist = p.drag * vx * abs(vx) + p.rolling * vx
        if vx > 1.0:
            resist += (1.0 - throttle) * p.engine_brake
        slope_fwd = -p.mass * GRAVITY * math.sin(self.pitch)     # bergauf bremst, bergab schiebt
        slope_side = p.mass * GRAVITY * math.sin(self.roll)      # Quergefälle schiebt nach rechts/links

        load = p.mass * GRAVITY * math.cos(self.pitch) + p.downforce * vx * vx
        shift = p.mass * self.accel * p.cg_height / L
        n_front = max(0.0, load * b / L - shift)
        n_rear = load * a / L

        if vx > 3.0:
            slip_front = math.atan2(vy + a * r, vx) - delta
            slip_rear = math.atan2(vy - b * r, vx)
            f_front = _tire_force(slip_front, p.stiffness_front, p.grip * n_front, p.slide_grip)
            rear_grip = p.rear_grip_factor + (p.drift_rear_grip - p.rear_grip_factor) * self.drift
            rear_slide = 1.0 + (p.drift_slide_grip - 1.0) * self.drift
            f_rear = _tire_force(slip_rear, p.stiffness_rear, p.grip * rear_grip * n_rear, rear_slide)
            ax = (drive - braking - resist + slope_fwd - f_front * math.sin(delta)) / p.mass + vy * r
            ay = (f_rear + f_front * math.cos(delta) + slope_side) / p.mass - vx * r
            r += (a * f_front * math.cos(delta) - b * f_rear) / p.inertia * dt
        else:
            ax = (drive - braking - resist + slope_fwd) / p.mass
            ay = 0.0
            vy *= math.exp(-10.0 * dt)
            r = vx * math.tan(delta) / L

        target_accel = (drive - braking - resist) / p.mass
        self.accel += (target_accel - self.accel) * (1.0 - math.exp(-dt / p.suspension_time))
        new_vx = vx + ax * dt
        if braking > 0 and new_vx < 0:
            new_vx = 0.0
        vx, vy = new_vx, vy + ay * dt

        self.yaw_rate = r
        self.heading = (self.heading + r * dt + math.pi) % (2 * math.pi) - math.pi
        fx, fz = self.forward
        rx, rz = self.right
        self.vx, self.vz = fx * vx + rx * vy, fz * vx + rz * vy

    def _down_accel(self):
        """Wie stark das Auto nach unten gezogen wird (m/s²): verstärkte Schwerkraft (Arcade, "schweres" Auto)
        + Abtrieb bei Tempo. Bestimmt, wann es über Kuppen abhebt und wie schnell es wieder landet."""
        p = self.p
        return GRAVITY * p.schwere + p.abtrieb_nach_unten * p.downforce * (self.vx ** 2 + self.vz ** 2) / p.mass

    def _fly(self, dt):
        """In der Luft: Schwerkraft, Luftwiderstand, Drehung klingt ab. Keine Reifenkräfte."""
        p = self.p
        speed = self.speed
        if speed > 0:
            k = p.drag * speed / p.mass * dt
            self.vx -= self.vx * k
            self.vz -= self.vz * k
        self.vy -= self._down_accel() * dt
        self.yaw_rate *= math.exp(-p.yaw_damping * dt)
        self.heading = (self.heading + self.yaw_rate * dt + math.pi) % (2 * math.pi) - math.pi

    def _move(self, surface, dt):
        """Bewegen, an Blockseiten abprallen, am Boden kleben oder abheben/landen."""
        p = self.p
        nx, nz = self.x + self.vx * dt, self.z + self.vz * dt
        down = self._down_accel()
        y_free = self.y + self.vy * dt - (0.0 if not self.grounded else 0.5 * down * dt * dt)
        top = max(self.y, y_free)

        # Blockseite im Weg? (Fahrbahn an der neuen Stelle viel höher als das Auto)
        g, sx, sz = surface.ground_slope(nx, nz)
        if g is not None and g > top + p.step_height:
            gx, gz = surface.ground(nx, self.z), surface.ground(self.x, nz)
            if gx is not None and gx > top + p.step_height:
                self.vx = -self.vx * p.abprall
                nx = self.x
            if gz is not None and gz > top + p.step_height:
                self.vz = -self.vz * p.abprall
                nz = self.z
            if nx != self.x and nz != self.z:      # nur diagonal blockiert
                self.vx, self.vz = -self.vx * p.abprall, -self.vz * p.abprall
                nx, nz = self.x, self.z
            self.hit = True
            g, sx, sz = surface.ground_slope(nx, nz)

        # Zu steile Fläche (z.B. oberer Teil des Streckenrands)? Dort haftet kein Auto: Sie wirkt wie eine
        # Wand. Das Auto prallt ab und gleitet entlang. Wer darüber fliegt (höher als die Fläche), ist frei.
        if g is not None and sx is not None and (self.grounded or y_free <= g):
            grad = math.hypot(sx, sz)
            ux, uz = (sx / grad, sz / grad) if grad > 0 else (0.0, 0.0)
            vn = self.vx * ux + self.vz * uz                 # Tempo quer in die Fläche hinein
            if grad > self._max_slope and 0 < vn < p.rand_durchbruch:   # viel zu schnell -> rampt drüber
                self.vx -= (1 + p.abprall) * vn * ux
                self.vz -= (1 + p.abprall) * vn * uz
                nx, nz = self.x, self.z
                self.hit = True
                g, sx, sz = surface.ground_slope(nx, nz)
        self.x, self.z = nx, nz

        if self.grounded:
            if g is not None and g >= y_free:
                if sx is not None:                   # der Fahrbahn folgen: Tempo nach oben = Steigung × Tempo
                    target = sx * self.vx + sz * self.vz          # (leicht geglättet, am Rand unbekannt)
                    target = max(-p.max_vy_boden, min(p.max_vy_boden, target))   # Rand/Rampe: kein Katapult
                    self.vy += (target - self.vy) * min(1.0, dt / VY_SMOOTH)
                self.y = g
            else:                                    # Fahrbahn fällt schneller weg als die Schwerkraft zieht:
                self.vy -= down * dt                 # das Auto folgt seiner Flugbahn ...
                self.y = y_free
                lift = SETTLE_LIFT if self.settle > 0 else LIFT
                if g is None or self.y - g > lift:   # ... und gilt erst bei deutlichem Abstand als abgehoben
                    self.grounded = False
                    self.airtime = 0.0
        else:
            self.y = y_free
            if g is not None and self.y <= g:
                self._land(g, surface)

    def _land(self, g, surface):
        p = self.p
        _, sx, sz = surface.ground_slope(self.x, self.z)
        if sx is None:                               # am Rand: Steigung in Fahrtrichtung abschätzen
            fx, fz = self.forward
            ahead = surface.ground(self.x + fx * 2.0, self.z + fz * 2.0)
            slope = (ahead - g) / 2.0 if ahead is not None and abs(ahead - g) < p.step_height else 0.0
            sx, sz = fx * slope, fz * slope
        along = sx * self.vx + sz * self.vz          # so schnell geht es entlang der Fahrbahn nach oben/unten
        impact = max(0.0, along - self.vy)           # Tempo senkrecht in die Fahrbahn hinein
        loss = p.landing_loss * min(1.0, impact / 10.0)
        self.vx *= 1.0 - loss
        self.vz *= 1.0 - loss
        self.y = g                                   # weiter entlang der Fahrbahn (auch auf Gefällen)
        self.vy = max(-p.max_vy_boden, min(p.max_vy_boden, sx * self.vx + sz * self.vz))
        self.grounded = True
        self.landed = impact
        self.pitch_rate = self.roll_rate = 0.0
        self.settle = SETTLE_TIME
