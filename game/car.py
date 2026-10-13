"""Fahrphysik eines Autos auf flacher Ebene ("Einspurmodell", wie in vielen Rennspielen).

Idee: Vorder- und Hinterreifen erzeugen Seitenkräfte. Bis zum Limit greifen sie fest,
lenkt man zu stark für das Tempo, reißt der Grip ab -> das Auto rutscht (untersteuert/driftet).

- Keine Grafik, kein Zufall: gleiche Eingaben -> exakt gleiche Bewegung.
- Fester Zeitschritt DT (1/60 s), intern in SUBSTEPS kleinere Schritte für Stabilität.
- Koordinaten (x, z), Heading h: Vorwärtsrichtung = (sin h, cos h). h=0 -> Norden (+z).
- Im Auto gedacht: vx = vorwärts, vy = nach rechts, yaw_rate > 0 = dreht nach rechts.

Alle Zahlenwerte stehen in config.toml. Wer dort etwas ändert, erhöht die Physik-Version
(Regel R8), damit alte Geister als "alte Version" erkennbar sind.
"""
import math

from .config import CONFIG

DT = 1.0 / CONFIG["simulation"]["physik_hz"]
SUBSTEPS = CONFIG["simulation"]["unterschritte"]
GRAVITY = CONFIG["simulation"]["gravitation"]


class CarParams:
    """Alle Fahrzeug-Werte aus config.toml (Abschnitte auto, antrieb, reifen, lenkung, drift).

    Beispiel: CarParams().grip  ->  1.8
    """

    SECTIONS = ("auto", "antrieb", "reifen", "lenkung", "drift")

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
    if value != value:   # NaN ist die einzige Zahl, die nicht gleich sich selbst ist
        return 0.0
    return max(low, min(high, value))


def _tire_force(slip, stiffness, max_force, slide_grip):
    """Seitenkraft eines Reifens in Abhängigkeit vom Schräglaufwinkel (vereinfachte "Magic Formula").

    Kleiner Winkel: Kraft steigt steil an (Reifen greift).
    Beim Grip-Limit: maximale Kraft.
    Darüber: Kraft fällt auf slide_grip ab -> der Reifen rutscht spürbar weg.
    """
    if max_force <= 0:
        return 0.0
    shape = 2.0 / math.pi * math.acos(-slide_grip) if slide_grip < 1 else 1.0   # z.B. 0.75 -> 1.46
    b = stiffness / (max_force * shape)
    return -max_force * math.sin(shape * math.atan(b * slip))


class Car:
    def __init__(self, x, z, heading, params=None):
        self.p = params or CarParams()
        self.x, self.z, self.heading = x, z, heading
        self.vx = self.vz = 0.0      # Geschwindigkeit in Weltkoordinaten
        self.yaw_rate = 0.0          # rad/s
        self.steer = 0.0             # aktuelle Lenkradstellung -1..1
        self.drift = 0.0             # Drift-Modus 0..1 (blendet weich ein und aus)
        self.accel = 0.0             # geglättete Längsbeschleunigung (für Gewichtsverlagerung)

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
        """(vorwärts, seitwärts) Geschwindigkeit in m/s, relativ zum Auto."""
        fx, fz = self.forward
        rx, rz = self.right
        return self.vx * fx + self.vz * fz, self.vx * rx + self.vz * rz

    def update(self, steer, throttle, brake, drift=0.0, dt=DT):
        """steer -1..1 (links..rechts), throttle 0..1, brake 0..1, drift 0..1 (Leertaste)."""
        steer, throttle = clip(steer, -1.0, 1.0), clip(throttle, 0.0, 1.0)
        brake, drift = clip(brake, 0.0, 1.0), clip(drift, 0.0, 1.0)
        # Das Lenkrad folgt der Eingabe nicht sofort, sondern mit begrenzter Geschwindigkeit
        step = self.p.steer_rate * dt
        self.steer += max(-step, min(step, steer - self.steer))
        step = self.p.drift_rate * dt
        self.drift += max(-step, min(step, drift - self.drift))
        for _ in range(SUBSTEPS):
            self._substep(throttle, brake, dt / SUBSTEPS)

    def _substep(self, throttle, brake, dt):
        p = self.p
        vx, vy = self.local_velocity()
        r = self.yaw_rate
        L, a, b = p.wheelbase, p.cg_to_front, p.cg_to_rear

        # Radeinschlag wird bei Tempo kleiner ("Lenkhilfe"): volle Lenkung bringt die Reifen
        # dann bei jedem Tempo ungefähr ans Grip-Limit, aber nicht weit darüber hinaus
        max_angle = p.max_steer / (1.0 + (vx / p.steer_speed_falloff) ** 2)
        if self.steer * r < 0:   # Gegenlenken (Lenkung entgegen der Drehung): im Drift voller Einschlag
            max_angle += (p.drift_max_steer - max_angle) * self.drift
        delta = self.steer * max_angle

        # --- Längskräfte: Motor, Bremse, Widerstand ---
        drive = throttle * min(p.max_drive_force, p.engine_power / max(vx, 1.0))
        braking = 0.0
        if brake > 0:
            if vx > 0.5:
                braking = brake * p.brake_force
            elif vx > -p.max_reverse:
                drive -= brake * p.reverse_force   # fast stehend: Bremse = rückwärts
        resist = p.drag * vx * abs(vx) + p.rolling * vx
        if vx > 1.0:
            resist += (1.0 - throttle) * p.engine_brake   # vom Gas gehen bremst leicht

        # --- Radlasten: Gewicht + Abtrieb, beim Bremsen mehr vorne ---
        load = p.mass * GRAVITY + p.downforce * vx * vx
        # (das Heck wird beim Bremsen absichtlich NICHT leichter -> kein Ausbrechen beim Bremsen)
        shift = p.mass * self.accel * p.cg_height / L
        n_front = max(0.0, load * b / L - shift)
        n_rear = load * a / L

        if vx > 3.0:
            # --- Reifenmodell: Schräglaufwinkel -> Seitenkraft ---
            slip_front = math.atan2(vy + a * r, vx) - delta
            slip_rear = math.atan2(vy - b * r, vx)
            f_front = _tire_force(slip_front, p.stiffness_front, p.grip * n_front, p.slide_grip)
            # Drift-Modus: Hinterachse von "mehr Grip als vorne" zu "weniger Grip" überblenden
            rear_grip = p.rear_grip_factor + (p.drift_rear_grip - p.rear_grip_factor) * self.drift
            rear_slide = 1.0 + (p.drift_slide_grip - 1.0) * self.drift
            f_rear = _tire_force(slip_rear, p.stiffness_rear, p.grip * rear_grip * n_rear, rear_slide)

            ax = (drive - braking - resist - f_front * math.sin(delta)) / p.mass + vy * r
            ay = (f_rear + f_front * math.cos(delta)) / p.mass - vx * r
            r += (a * f_front * math.cos(delta) - b * f_rear) / p.inertia * dt
        else:
            # Sehr langsam oder rückwärts: einfach rollen wie ein Spielzeugauto (stabil beim Rangieren)
            ax = (drive - braking - resist) / p.mass
            ay = 0.0
            vy *= math.exp(-10.0 * dt)
            r = vx * math.tan(delta) / L

        # Federung: das Gewicht verlagert sich nicht sofort, sondern mit kurzer Verzögerung
        target_accel = (drive - braking - resist) / p.mass
        self.accel += (target_accel - self.accel) * (1.0 - math.exp(-dt / p.suspension_time))

        new_vx = vx + ax * dt
        if braking > 0 and new_vx < 0:   # Bremse hält an, fährt aber nicht rückwärts
            new_vx = 0.0
        vx, vy = new_vx, vy + ay * dt

        # --- Drehen und bewegen ---
        self.yaw_rate = r
        self.heading = (self.heading + r * dt + math.pi) % (2 * math.pi) - math.pi
        fx, fz = self.forward
        rx, rz = self.right
        self.vx, self.vz = fx * vx + rx * vy, fz * vx + rz * vy
        self.x += self.vx * dt
        self.z += self.vz * dt
