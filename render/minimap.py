"""Navi-Karte oben rechts: Strecke von oben, dreht sich mit dem Auto (Fahrtrichtung = oben).

Die Karte liegt in Ursinas Bildschirm-Ebene (wie das HUD). Ein "Scissor" schneidet alles ab,
was außerhalb des Kartenfensters liegt.

Aufbau:  frame (Fenster, fest)  ->  pivot (dreht sich, Drehpunkt = Auto)  ->  map (Strecke in Metern)
"""
import math

from panda3d.core import Point3
from ursina import Entity, Mesh, Vec3, camera, color, window
from ursina.shaders import unlit_shader

VIEW = 350.0          # so viele Meter Strecke passen in das Fenster
LOOKAHEAD = 0.28      # Auto sitzt so weit unter der Mitte -> man sieht mehr nach vorne
SIZE = 0.26           # Kantenlänge als Anteil der Bildschirmhöhe
MARGIN = 0.025        # Abstand zum Bildschirmrand
WIDEN = 1.3           # Strecke etwas breiter zeichnen, damit sie auf der kleinen Karte lesbar ist

ROAD = color.rgba(0.82, 0.84, 0.9, 0.85)
CHECKPOINT = color.rgba(1.0, 0.8, 0.15, 1)
FINISH = color.rgba(1, 1, 1, 1)
CAR = color.rgba(1.0, 0.3, 0.15, 1)
BACKGROUND = color.rgba(0.04, 0.06, 0.1, 0.72)
BORDER = color.rgba(1, 1, 1, 0.35)


def _band(left, right, indices):
    """Streifen zwischen linkem und rechtem Rand über die gegebenen Punkt-Indizes."""
    verts, tris = [], []
    for k, i in enumerate(indices):
        verts += [Vec3(left[i][0], left[i][1], 0), Vec3(right[i][0], right[i][1], 0)]
        if k > 0:
            a = 2 * k - 2
            tris += [a, a + 2, a + 1, a + 1, a + 2, a + 3]
    return Mesh(vertices=verts, triangles=tris)


def _bar(gate, width):
    """Querbalken über die Strecke (Checkpoint / Ziel)."""
    tx, tz = gate.tangent
    nx, nz = -tz, tx
    cx, cz = gate.center
    h, d = width / 2 + 2, 2.5
    pts = [(cx + nx * h - tx * d, cz + nz * h - tz * d), (cx + nx * h + tx * d, cz + nz * h + tz * d),
           (cx - nx * h + tx * d, cz - nz * h + tz * d), (cx - nx * h - tx * d, cz - nz * h - tz * d)]
    return Mesh(vertices=[Vec3(x, z, 0) for x, z in pts], triangles=[0, 1, 2, 0, 2, 3])


def _shape(points):
    """Kleine Fläche aus Eckpunkten (Pfeil, Raute)."""
    verts = [Vec3(x, y, 0) for x, y in points]
    return Mesh(vertices=verts, triangles=[0, 1, 2, 0, 2, 3])


class Minimap:
    def __init__(self, track, ghost_colors=()):
        self.track = track
        self.k = SIZE / VIEW          # Meter -> Bildschirm-Einheiten

        def add(parent, model, col, layer, **kw):
            """layer = Zeichenreihenfolge (größer = weiter oben)."""
            e = Entity(parent=parent, model=model, color=col, shader=unlit_shader, double_sided=True, **kw)
            e.setBin("fixed", 100 + layer)
            e.setDepthTest(False)
            e.setDepthWrite(False)
            return e

        self.frame = Entity(parent=camera.ui)
        half = SIZE / 2
        self.frame.setScissor(Point3(-half, -half, 0), Point3(half, half, 0))
        add(self.frame, "quad", BORDER, 0, scale=SIZE + 0.006)
        add(self.frame, "quad", BACKGROUND, 1, scale=SIZE)

        car_y = -SIZE * LOOKAHEAD
        self.pivot = Entity(parent=self.frame, y=car_y)
        self.map = Entity(parent=self.pivot, scale=self.k)

        c, l, r = track.center, track.left_edge, track.right_edge
        self.left = c + (l - c) * WIDEN
        self.right = c + (r - c) * WIDEN
        add(self.map, _band(self.left, self.right, range(len(c))), ROAD, 2)
        for g in track.checkpoints:
            add(self.map, _bar(g, track.width * WIDEN), CHECKPOINT, 4)
        add(self.map, _bar(track.start_gate, track.width * WIDEN), FINISH, 4)
        if not track.is_loop:
            add(self.map, _bar(track.finish_gate, track.width * WIDEN), FINISH, 4)
        self.ghost_dots = [add(self.map, _shape([(0, 10), (-10, 0), (0, -10), (10, 0)]), gc, 5)
                           for gc in ghost_colors]
        # Auto: Pfeil, zeigt immer nach oben (die Karte dreht sich drumherum)
        add(self.frame, _shape([(0, 16), (-10, -11), (0, -5), (10, -11)]), CAR, 6, y=car_y, scale=self.k)
        self._aspect = None
        self.right = None             # rechter Rand in UI-Einheiten (None = Fensterrand); viewer.py setzt ihn
        self.top = 0.5
        self._placed = None

    def _place(self):
        """Oben rechts in die Ecke setzen (neu, falls sich Fenster oder Ränder ändern)."""
        key = (window.aspect_ratio, self.right, self.top)
        if key == self._placed:
            return
        self._placed = key
        self._aspect = window.aspect_ratio
        right = self._aspect / 2 if self.right is None else self.right
        self.frame.position = Vec3(right - MARGIN - SIZE / 2, self.top - MARGIN - SIZE / 2, 0)

    def update(self, x, z, heading, ghost_positions=()):
        self._place()
        self.map.position = Vec3(-x * self.k, -z * self.k, 0)
        self.pivot.rotation_z = -math.degrees(heading)  # Karte so drehen, dass die Fahrtrichtung oben ist
        for dot, (gx, gz) in zip(self.ghost_dots, ghost_positions):
            dot.position = Vec3(gx, gz, 0)
