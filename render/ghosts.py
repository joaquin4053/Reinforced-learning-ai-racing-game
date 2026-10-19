"""Viele Versuche gleichzeitig als Geister (Plan-Schritt 7) für viewer.py.

Damit bis zu 100 Geister flüssig laufen, ist jeder Geist ein einfaches Modell aus zwei Quadern in einem einzigen
Mesh (ein Zeichenaufruf pro Geist, statt des 3D-Autos), halb durchsichtig, ohne Schatten und ohne Kollision.
Farbe nach der aktuellen Position im Rennen: vorderes Drittel grün, mittleres orange, hinteres rot.
Ein gecrashter Geist bleibt an der Crash-Stelle stehen und wird grau. Über jedem Geist steht seine Nummer
(ausblendbar). Der Geist im Fokus wird nicht hier gezeichnet, sondern als richtiges Auto (World.car).
"""
import math

from ursina import Entity, Mesh, Vec3, camera, color
from ursina.shaders import unlit_shader

from render.hud import TextQuad

MAX_GHOSTS = 100
ALPHA = 0.5
NEAR = (5.0, 14.0)          # m zur Kamera: näher als 5 m unsichtbar, ab 14 m voll sichtbar (verdeckt sonst das Bild)
LABEL_RANGE = 120.0         # m: weiter weg keine Nummer (sonst ballen sie sich am Horizont)
MAX_LABELS = 30             # höchstens so viele Nummern (die nächsten), mehr sind nicht lesbar
GREY = (0.55, 0.55, 0.6)


def _box(cx, cy, cz, sx, sy, sz, shade, verts, tris, cols):
    """Quader (Mitte, Grösse) an die Listen anhängen; shade = Helligkeit (Vertex-Farbe)."""
    o = len(verts)
    for dx in (-.5, .5):
        for dy in (-.5, .5):
            for dz in (-.5, .5):
                verts.append((cx + dx * sx, cy + dy * sy, cz + dz * sz))
                cols.append(color.rgba(shade, shade, shade, 1))
    for a, b, c, d in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
        tris += [o + a, o + b, o + c, o + a, o + c, o + d]


def ghost_mesh():
    """Karosserie + Dach in einem Mesh (das Dach etwas dunkler)."""
    verts, tris, cols = [], [], []
    _box(0, 0.6, 0, 1.9, 0.7, 4.3, 1.0, verts, tris, cols)
    _box(0, 1.2, -0.35, 1.5, 0.55, 2.0, 0.75, verts, tris, cols)
    return Mesh(vertices=verts, triangles=tris, colors=cols)


THIRDS = ((0.25, 0.9, 0.35), (1.0, 0.58, 0.1), (0.95, 0.22, 0.18))   # grün, orange, rot


def rank_rgb(place, n):
    """Farbe nach Platz (1 = vorne) in einer Gruppe von n: vorderes Drittel grün, mittleres orange, hinteres rot."""
    if place is None or n <= 1:
        return THIRDS[1]
    return THIRDS[min(2, (place - 1) * 3 // n)]


class SimpleGhost:
    def __init__(self):
        self.root = Entity()
        self.tilt = Entity(parent=self.root)
        self.body = Entity(parent=self.tilt, model=ghost_mesh(), shader=unlit_shader, double_sided=True)
        self.root.hide(0b0001)                          # kein Schatten (Geist, nur Anzeige)
        self.label = TextQuad(parent=camera.ui, height=0.02, origin=(0, -.5), heavy=True, px=40)
        self.versuch = None
        self.ui = None                                   # Position auf dem Bildschirm (für Klick und Nummer)
        self._style = None
        self._parked = None                              # (Versuch) steht still: Pose nicht jedes Bild neu setzen

    def show(self, on):
        if self.root.enabled != on:                      # Ein-/Ausschalten ist in Ursina teuer: nur bei Änderung
            self.root.enabled = on
        if not on:
            self.set_label(False)
            self.ui = None

    def set_label(self, on):
        if self.label.enabled != on:
            self.label.enabled = on

    def set_pose(self, x, y, z, heading, pitch, roll):
        self.root.position = Vec3(x, y, z)
        self.root.rotation_y = math.degrees(heading)
        self.tilt.rotation_x = -math.degrees(pitch)
        self.tilt.rotation_z = math.degrees(roll)

    def set_style(self, rgb, crashed, text, label_rgb, fade=1.0):
        fade = round(fade, 2)
        style = (rgb, crashed, text, fade)
        if style == self._style:
            return
        self._style = style
        if crashed:                                      # an der Crash-Stelle, leicht ausgegraut
            rgb = tuple(c * 0.35 + g * 0.65 for c, g in zip(rgb, GREY))
            alpha = 0.32
        else:
            alpha = ALPHA
        alpha *= fade
        if self.tilt.enabled != (fade > 0.02):
            self.tilt.enabled = fade > 0.02
        self.body.color = color.rgba(*rgb, alpha)
        self.label.set(text, label_rgb)


class GhostFleet:
    """Alle Geister einer Gruppe. update() setzt Pose, Farbe und Nummer, pick() findet einen angeklickten."""

    def __init__(self):
        self.cars = [SimpleGhost() for _ in range(MAX_GHOSTS)]
        for c in self.cars:
            c.show(False)
        self.labels_on = True

    def update(self, group, focus, t, to_ui, cam_pos=None, places=None):
        """group = [(Attempt, Episode)] in Versuchs-Reihenfolge, focus = Versuchsnummer des Fokus,
        places = {Versuch: Platz} aus der Rangliste (bestimmt die Farbe).
        to_ui(Entity-Position) -> (x, y) im UI oder None (hinter der Kamera / ausserhalb der 3D-Ansicht)."""
        others = [(i, a, ep) for i, (a, ep) in enumerate(group) if a.versuch != focus and ep is not None]
        n = len(group)
        for car, (i, a, ep) in zip(self.cars, others[:MAX_GHOSTS]):
            car.show(True)
            ended = t >= ep.duration
            if car.versuch != a.versuch or not ended or car._parked != a.versuch:
                x, y, z, heading, pitch, roll, _ = ep.pose(t)
                car.set_pose(x, y, z, heading, pitch, roll)
                car._parked = a.versuch if ended else None
            else:                                        # steht schon (Crash oder Ziel): nur lesen
                x, y, z = car.root.x, car.root.y, car.root.z
            car.versuch = a.versuch
            crashed = ended and a.ergebnis == "Crash"
            rgb = rank_rgb((places or {}).get(a.versuch), n)
            text = f"{a.versuch:,}".replace(",", "'") + (" Crash" if crashed else " Ziel" if ended and a.gueltig else "")
            label_rgb = (170, 170, 180) if crashed else tuple(int(80 + 175 * c) for c in rgb)
            fade, d = 1.0, 0.0
            if cam_pos is not None:
                d = math.dist((x, y, z), (cam_pos.x, cam_pos.y, cam_pos.z))
                fade = max(0.0, min(1.0, (d - NEAR[0]) / (NEAR[1] - NEAR[0])))
            car.set_style(rgb, crashed, text, label_rgb, fade)
            above = to_ui(Vec3(x, y + 2.0, z))          # eine Umrechnung für Klick und Nummer
            car.ui = above
            car.wants_label = self.labels_on and above is not None and d < LABEL_RANGE and fade > 0.3
            car.dist = d
        for car in self.cars[len(others):]:
            car.show(False)
        active = self.cars[:len(others)]
        near = set(id(c) for c in sorted((c for c in active if c.wants_label), key=lambda c: c.dist)[:MAX_LABELS])
        for car in active:
            on = id(car) in near
            car.set_label(on)
            if on:
                car.label.position = Vec3(car.ui[0], car.ui[1], 0)

    def pick(self, mx, my, radius=0.045):
        """Versuchsnummer des Geists, der dem Mausklick am nächsten ist (oder None)."""
        best, best_d = None, radius
        for car in self.cars:
            if car.root.enabled and car.ui is not None:
                d = math.hypot(car.ui[0] - mx, car.ui[1] - my)
                if d < best_d:
                    best, best_d = car.versuch, d
        return best

    def toggle_labels(self):
        self.labels_on = not self.labels_on
