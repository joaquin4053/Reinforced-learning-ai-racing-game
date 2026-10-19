"""Auto-Modell aus einer 3D-Datei (assets/autos/...), mit drehenden Rädern und lenkenden Vorderrädern.

Aktuell: "2020 NASCAR Chevrolet Camaro Zl1 1LE" von Ddiaz Design, CC-BY-4.0 (siehe CREDITS.md und
assets/autos/camaro/license.txt). Nur Anzeige: Die Physik rechnet weiter mit den Werten aus config.toml.

Das Modell wird so skaliert und verschoben, dass seine Achsen dort sitzen, wo die Physik sie rechnet
(cg_to_front / cg_to_rear). Fehlt die Datei, nimmt das Spiel den eingebauten Low-Poly-Buggy.
"""
import math
from pathlib import Path

from panda3d.core import CullFaceAttrib, Filename, Loader, NodePath, TransparencyAttrib
from ursina import Entity, Vec3

from game.car import CarParams

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "assets" / "autos" / "camaro" / "camaro.bam"

# Masse der Datei (in Metern nach Skalierung 100, gemessen beim Einbau):
SOURCE_SCALE = 100.0            # Datei ist in Zentimetern gebaut
SOURCE_FRONT_Y = -1.494         # Vorderachse (in der Datei zeigt die Front nach -y)
SOURCE_REAR_Y = 1.276           # Hinterachse
WHEEL_RADIUS_SOURCE = 0.354     # m
WHEELS = {"3DWheel Front L": True, "3DWheel Front R": True, "3DWheel Rear L": False, "3DWheel Rear R": False}
CALLIPERS = {"Calliper Front L": "3DWheel Front L", "Calliper Front R": "3DWheel Front R"}

_cache = {}


def available():
    return MODEL_FILE.exists()


def _load():
    if "model" not in _cache:
        _cache["model"] = NodePath(Loader.get_global_ptr().load_sync(Filename.from_os_specific(str(MODEL_FILE))))
    return _cache["model"]


class ModelCar(Entity):
    """Gleiche Schnittstelle wie render.world.CarModel: set_state(x, y, z, heading, pitch, roll, steer)."""

    def __init__(self, body_color=None, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        p = CarParams()
        self.body = Entity(parent=self)
        holder = Entity(parent=self.body)
        # Grösse: Radstand der Datei -> Radstand der Physik
        wheelbase = SOURCE_REAR_Y - SOURCE_FRONT_Y
        k = p.wheelbase / wheelbase
        self.wheel_radius = WHEEL_RADIUS_SOURCE * k
        model = _load().copy_to(holder)
        # Ursina ordnet die Achsen anders an als die Datei -> ohne Spiegeln wäre die Schrift seitenverkehrt.
        model.set_scale(-SOURCE_SCALE * k, SOURCE_SCALE * k, SOURCE_SCALE * k)
        model.set_attrib(CullFaceAttrib.make_reverse())        # gespiegelt: Vorder-/Rückseiten tauschen
        # Datei: z oben, Front bei -y. Ursina: y oben, vorne +z. Um +90° kippen (im Bild geprüft).
        model.set_hpr(0, 90, 0)
        # Achsen an die Stellen der Physik schieben (Vorderachse cg_to_front vor dem Schwerpunkt)
        front = -SOURCE_FRONT_Y * k
        model.set_pos(0, 0, p.cg_to_front - front)
        self.model_np = model

        # Räder: jedes an einen Drehpunkt in seiner Mitte hängen (lenken = steer, rollen = spin)
        self.wheels = []
        for name, steerable in WHEELS.items():
            w = model.find(f"**/{name}")
            if w.is_empty():
                continue
            a, b = w.get_tight_bounds(holder)
            center = (a + b) / 2
            steer = Entity(parent=holder, position=Vec3(center[0], center[1], center[2]))
            spin = Entity(parent=steer)
            w.wrt_reparent_to(spin)
            if steerable:
                cal = model.find(f"**/{[c for c, wn in CALLIPERS.items() if wn == name][0]}")
                if not cal.is_empty():
                    cal.wrt_reparent_to(steer)                     # Bremssattel lenkt mit, dreht sich nicht
            self.wheels.append((steer, spin, steerable))

        if alpha < 1.0:                                            # Geist: halbdurchsichtig, eingefärbt
            self.set_transparency(TransparencyAttrib.M_alpha)
            c = body_color
            self.set_color_scale(c.r if c else 1, c.g if c else 1, c.b if c else 1, alpha)
        self._spin = 0.0
        self._last = None

    def set_state(self, x, y, z, heading, pitch=0.0, roll=0.0, steer=0.0):
        if self._last is not None:                                  # Räder rollen: Weg / Radius
            lx, lz = self._last
            d = (x - lx) * math.sin(heading) + (z - lz) * math.cos(heading)
            if abs(d) < 20:
                self._spin = (self._spin + math.degrees(d / self.wheel_radius)) % 360
        self._last = (x, z)
        self.position = Vec3(x, y, z)
        self.rotation_y = math.degrees(heading)
        self.body.rotation_x = -math.degrees(pitch)
        self.body.rotation_z = math.degrees(roll)       # Ursina: + = rechte Seite tiefer (geprüft im Bild)
        for steer_np, spin_np, steerable in self.wheels:
            spin_np.rotation_x = self._spin             # um die Achse (x) rollen
            if steerable:
                steer_np.rotation_y = steer * 28        # wie beim Buggy: + = nach rechts
