"""Zeitleiste fürs Zuschauen: Schieberegler über alle gespeicherten Versionen eines Trainings.

Ganz rechts = LIVE (immer die neueste Version). Mit der Maus ziehen oder mit ←/→ schrittweise wählen;
beim Loslassen wird die gewählte Version geladen. Gebaut aus eigenen Text-Bausteinen (Ursinas eigener
Text funktioniert im macOS-Core-Profil nicht).
"""
from ursina import Entity, Vec3, camera, color, mouse
from ursina.shaders import unlit_shader

from render.hud import TextQuad


class Timeline:
    BAR_W, BAR_H = 0.95, 0.016
    Y = -0.255

    def __init__(self):
        self.root = Entity(parent=camera.ui, position=Vec3(0, self.Y, 0))
        self.back = Entity(parent=self.root, model="quad", color=color.rgba(0.07, 0.09, 0.2, 0.78),
                           scale=(self.BAR_W + 0.06, 0.075), shader=unlit_shader, z=0.02)
        self.bar = Entity(parent=self.root, model="quad", color=color.rgba(0.55, 0.6, 0.75, 0.9),
                          scale=(self.BAR_W, self.BAR_H), shader=unlit_shader, z=0.01)
        self.fill = Entity(parent=self.root, model="quad", color=color.rgb32(255, 170, 40), origin=(-.5, 0),
                           x=-self.BAR_W / 2, scale=(0.0, self.BAR_H), shader=unlit_shader)
        self.knob = Entity(parent=self.root, model="quad", color=color.white, scale=(0.018, 0.045),
                           shader=unlit_shader, z=-0.01)
        self.label = TextQuad(parent=self.root, position=(0, 0.06, 0), height=0.026, origin=(0, -.5), heavy=True)
        self.hint = TextQuad("Zeitleiste: ziehen oder  ← / →   ·   L = live   ·   Z = ausblenden",
                             parent=self.root, position=(0, -0.04, 0), height=0.02, origin=(0, .5))
        self.steps = []          # Schritte jeder Version, aufsteigend
        self.index = 0           # gewählte Version (len(steps)-1 = neueste)
        self.live = True         # True = immer die neueste
        self.dragging = False

    # ---------- Daten ----------
    def set_versions(self, steps):
        self.steps = list(steps)
        if self.live or self.index >= len(self.steps):
            self.index = max(0, len(self.steps) - 1)
        self._draw()

    @property
    def selected(self):
        return self.steps[self.index] if self.steps else None

    # ---------- Bedienung ----------
    def step(self, delta):
        """Eine Version weiter (+1) oder zurück (-1). Rechts am Ende = wieder live."""
        if not self.steps:
            return
        self.index = max(0, min(len(self.steps) - 1, self.index + delta))
        self.live = delta > 0 and self.index == len(self.steps) - 1
        self._draw()

    def go_live(self):
        self.live = True
        self.index = max(0, len(self.steps) - 1)
        self._draw()

    @property
    def visible(self):
        return self.root.enabled

    def toggle(self):
        """Leiste ein- oder ausblenden (Taste Z)."""
        self.root.enabled = not self.root.enabled
        self.dragging = False

    def hovered(self):
        if not self.visible:
            return False
        mx, my = mouse.x, mouse.y
        return abs(mx - self.root.x) <= self.BAR_W / 2 + 0.03 and abs(my - self.Y) <= 0.04

    def drag_to_mouse(self):
        if not self.steps:
            return
        t = (mouse.x - self.root.x + self.BAR_W / 2) / self.BAR_W
        t = max(0.0, min(1.0, t))
        self.index = round(t * (len(self.steps) - 1))
        self.live = self.index == len(self.steps) - 1
        self._draw()

    # ---------- Anzeige ----------
    def label_text(self, info=None):
        if not self.steps:
            return "noch keine Versionen"
        nr = f"Version {self.index + 1}/{len(self.steps)} · Schritt {self.selected:,}".replace(",", "'")
        if self.live:
            nr = "LIVE · " + nr
        return nr + (f"  ·  {info}" if info else "")

    def _draw(self, info=None):
        n = len(self.steps)
        t = self.index / (n - 1) if n > 1 else 1.0
        self.knob.x = -self.BAR_W / 2 + t * self.BAR_W
        self.fill.scale_x = t * self.BAR_W
        self.knob.color = color.rgb32(255, 80, 80) if self.live else color.white
        self.label.text = self.label_text(info)

    def show_info(self, info):
        self.label.text = self.label_text(info)
