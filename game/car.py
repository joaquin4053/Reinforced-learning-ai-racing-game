"""Fahrphysik eines Autos auf flacher Ebene ("Einspurmodell", wie in vielen Rennspielen).

Idee: Vorder- und Hinterreifen erzeugen Seitenkräfte. Bis zum Limit greifen sie fest,
lenkt man zu stark für das Tempo, reißt der Grip ab -> das Auto rutscht (untersteuert/driftet).

- Keine Grafik, kein Zufall: gleiche Eingaben -> exakt gleiche Bewegung.
- Fester Zeitschritt DT (1/60 s), intern in SUBSTEPS kleinere Schritte für Stabilität.
- Koordinaten (x, z), Heading h: Vorwärtsrichtung = (sin h, cos h). h=0 -> Norden (+z).
- Im Auto gedacht: vx = vorwärts, vy = nach rechts, yaw_rate > 0 = dreht nach rechts.

Wenn du hier Werte änderst, erhöhe PHYSICS_VERSION in sim.py,
damit alte Geister als "alte Version" erkennbar sind.
"""
import math

DT = 1.0 / 60.0
SUBSTEPS = 4
GRAVITY = 9.81


class CarParams:
    # Auto
    mass = 1200.0            # kg
    inertia = 2100.0         # kg·m² Drehträgheit (größer = dreht ruhiger ein, aber träger)
    cg_to_front = 1.25       # m Schwerpunkt -> Vorderachse
    cg_to_rear = 1.35        # m Schwerpunkt -> Hinterachse
    cg_height = 0.30         # m Schwerpunkthöhe: beim Bremsen mehr Last vorne (besseres Einlenken)
    suspension_time = 0.15   # s so lange braucht die Federung, bis sich das Gewicht verlagert hat
    length = 4.0
    width = 2.0

    # Antrieb, Bremse, Widerstand
    engine_power = 240_000.0     # W  (ca. 325 PS)
    max_drive_force = 11_000.0   # N  Traktionsgrenze beim Anfahren
    brake_force = 16_000.0       # N  bei voller Bremse
    reverse_force = 4_000.0      # N  Bremse im Stand = rückwärts
    max_reverse = 8.0            # m/s
    drag = 0.95                  # Luftwiderstand  F = drag · v²  -> Topspeed ca. 220 km/h
    rolling = 40.0               # Rollwiderstand  F = rolling · v
    engine_brake = 1800.0        # N  Motorbremse, wenn man vom Gas geht
    downforce = 1.2              # Abtrieb         F = downforce · v²  (etwas mehr Grip bei Tempo)

    # Reifen
    grip = 1.8                   # Reibwert μ (1.0 = Straßenreifen, 1.5-1.8 = Rennreifen)
    stiffness_front = 150_000.0  # N/rad wie direkt der Reifen greift (höher = präziser, weniger Schwimmen)
    stiffness_rear = 200_000.0   # hinten steifer -> Auto ist stabil, leicht untersteuernd
    slide_grip = 0.85            # Anteil Grip, der vorne bleibt, wenn der Reifen rutscht (über dem Limit)
    rear_grip_factor = 1.5       # Hinterachse hat immer mehr Grip als vorne -> Auto übersteuert nie,
                                 # sondern schiebt bei zu viel Tempo über die Vorderräder (untersteuert)

    # Lenkung
    max_steer = 0.50             # rad maximaler Radeinschlag im Stand (~29°)
    steer_speed_falloff = 17.0   # m/s: bei diesem Tempo nur noch halber Einschlag (wird schnell kleiner)
    steer_rate = 5.0             # wie schnell das Lenkrad folgt (1/s, 5 = in 0.2 s von Mitte auf Anschlag)

    # Drift-Modus (Leertaste): Heck verliert Grip, mehr Lenkeinschlag zum Gegensteuern
    drift_rear_grip = 0.75       # Grip hinten im Drift (statt rear_grip_factor), < 1 -> Heck bricht aus
    drift_slide_grip = 0.85      # Hinterreifen rutschen im Drift mit diesem Anteil Grip weiter
    drift_max_steer = 0.60       # rad Lenkeinschlag beim Gegenlenken im Drift, bei jedem Tempo (~35°)
    drift_rate = 5.0             # wie schnell der Drift-Modus ein-/ausblendet (1/s)

    @property
    def wheelbase(self):
        return self.cg_to_front + self.cg_to_rear


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
        steer = max(-1.0, min(1.0, steer))
        throttle = max(0.0, min(1.0, throttle))
        brake = max(0.0, min(1.0, brake))
        # Das Lenkrad folgt der Eingabe nicht sofort, sondern mit begrenzter Geschwindigkeit
        step = self.p.steer_rate * dt
        self.steer += max(-step, min(step, steer - self.steer))
        drift = max(0.0, min(1.0, drift))
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
