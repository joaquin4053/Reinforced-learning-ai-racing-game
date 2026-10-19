"""Bedienelemente für viewer.py: Info-Zeile, zwei Zeitleisten, Knöpfe, Panel, Hilfe. (Plan-Schritt 6)

Alles liegt in camera.ui (gleiches Fenster, gleiche Bildschleife wie die 3D-Ansicht) und wird aus eigenen
Quads und TextQuad gebaut (Ursinas eigener Text funktioniert im macOS-Core-Profil nicht).
Koordinaten wie in Ursina: y von -0.5 (unten) bis 0.5 (oben), x von -A/2 bis A/2 (A = Seitenverhältnis).
Layout.apply() setzt alles neu, wenn sich Fenstergrösse oder Panel ändern.
"""
import numpy as np
from PIL import Image, ImageDraw
from panda3d.core import SamplerState
from ursina import Entity, Texture, Vec3, camera, color
from ursina.shaders import unlit_shader

from render.hud import TextLines, TextQuad

BAR = color.rgba(0.07, 0.09, 0.2, 0.92)
STRIP_BG = color.rgba(0.02, 0.03, 0.08, 1)
BUTTON = color.rgba(0.16, 0.19, 0.34, 1)
BUTTON_HOVER = color.rgba(0.26, 0.31, 0.52, 1)
BUTTON_ON = color.rgb32(220, 60, 50)
WHITE, GREY, YELLOW = (255, 255, 255), (170, 178, 200), (255, 215, 40)

# Farben in der Versuchs-Zeitleiste
C_ZIEL = (70, 220, 110)
C_ZIEL_AUSWERTUNG = (255, 205, 40)
C_CRASH = (215, 70, 60)
C_ABBRUCH = (140, 140, 150)


def _quad(parent, col, z=0.0, **kw):
    return Entity(parent=parent, model="quad", color=col, shader=unlit_shader, z=z, **kw)


def _texture(img, nearest=False):
    tex = Texture(img)
    f = SamplerState.FT_nearest if nearest else SamplerState.FT_linear
    tex._texture.setMinfilter(f)
    tex._texture.setMagfilter(f)
    return tex


def _place(e, x0, y0, x1, y1):
    """Quad über das Rechteck (x0, y0) .. (x1, y1) legen."""
    e.position = Vec3((x0 + x1) / 2, (y0 + y1) / 2, e.z)
    e.scale = Vec3(max(x1 - x0, 1e-4), max(y1 - y0, 1e-4), 1)


# ---------- Symbole für die Knöpfe (selbst gezeichnet, Schriften haben sie nicht sicher) ----------
def _icon(kind, px=64):
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w = (255, 255, 255, 255)
    m = px * 0.2
    if kind == "play":
        d.polygon([(m * 1.3, m), (px - m, px / 2), (m * 1.3, px - m)], fill=w)
    elif kind == "pause":
        d.rectangle([m * 1.2, m, px * 0.42, px - m], fill=w)
        d.rectangle([px * 0.58, m, px - m * 1.2, px - m], fill=w)
    elif kind in ("prev", "next"):
        pts = [(px * 0.62, m), (px * 0.22, px / 2), (px * 0.62, px - m)]
        bar = [px * 0.14, m, px * 0.26, px - m]
        if kind == "next":
            pts = [(px - x, y) for x, y in pts]
            bar = [px - bar[2], m, px - bar[0], px - m]
        d.polygon(pts, fill=w)
        d.rectangle(bar, fill=w)
    return _texture(img)


class Button:
    def __init__(self, label="", icon=None, on_click=None, tooltip=""):
        self.root = Entity(parent=camera.ui)
        self.bg = _quad(self.root, BUTTON, z=0.01)
        self.icon = _quad(self.root, color.white, texture=_icon(icon)) if icon else None
        self.text = TextQuad(label, parent=self.root, height=0.022, origin=(0, 0), heavy=True, px=44) if not icon else None
        self.on_click, self.tooltip = on_click, tooltip
        self.rect = (0, 0, 0, 0)
        self.active = False
        self._icon_name = icon

    def set_icon(self, name):
        if self.icon and name != self._icon_name:
            self._icon_name = name
            self.icon.texture = _icon(name)

    def set_label(self, text):
        if self.text:
            self.text.text = text

    def layout(self, x0, y0, x1, y1):
        self.rect = (x0, y0, x1, y1)
        _place(self.bg, x0, y0, x1, y1)
        cx, cy, h = (x0 + x1) / 2, (y0 + y1) / 2, y1 - y0
        if self.icon:
            self.icon.position = Vec3(cx, cy, 0)
            self.icon.scale = Vec3(h * 0.8, h * 0.8, 1)
        if self.text:
            self.text.position = Vec3(cx, cy, 0)

    def contains(self, mx, my):
        x0, y0, x1, y1 = self.rect
        return x0 <= mx <= x1 and y0 <= my <= y1

    def update(self, mx, my):
        self.bg.color = BUTTON_ON if self.active else BUTTON_HOVER if self.contains(mx, my) else BUTTON


class Strip:
    """Zeitleiste: Bild (Balken) + Knopf an der aktuellen Stelle. frac = 0 (links) .. 1 (rechts)."""

    def __init__(self, title):
        self.root = Entity(parent=camera.ui)
        self.bg = _quad(self.root, STRIP_BG, z=0.02)
        self.image = _quad(self.root, color.white, z=0.01)
        self.played = _quad(self.root, color.rgba(1, 1, 1, 0.10), z=0.005)
        self.knob = _quad(self.root, color.white, z=-0.01)
        self.hover_line = _quad(self.root, color.rgba(1, 1, 1, 0.5), z=-0.005)
        self.title = TextQuad(title, parent=self.root, height=0.019, origin=(-.5, 0), px=40, color=GREY)
        self.right = TextQuad("", parent=self.root, height=0.019, origin=(-.5, 0), px=40)
        self.rect = (0, 0, 1, 1)
        self.frac = 0.0
        self.dragging = False

    def layout(self, x_label, x0, y0, x1, y1, x_right):
        self.rect = (x0, y0, x1, y1)
        _place(self.bg, x0 - 0.004, y0 - 0.004, x1 + 0.004, y1 + 0.004)
        _place(self.image, x0, y0, x1, y1)
        self.title.position = Vec3(x_label, (y0 + y1) / 2, 0)
        self.right.position = Vec3(x_right, (y0 + y1) / 2, 0)
        self.set_frac(self.frac)

    def set_image(self, img):
        self.image.texture = _texture(img, nearest=True)

    def set_frac(self, frac):
        self.frac = max(0.0, min(1.0, frac))
        x0, y0, x1, y1 = self.rect
        x = x0 + (x1 - x0) * self.frac
        _place(self.knob, x - 0.0035, y0 - 0.008, x + 0.0035, y1 + 0.008)
        _place(self.played, x0, y0, x, y1)

    def contains(self, mx, my, pad=0.012):
        x0, y0, x1, y1 = self.rect
        return x0 - pad <= mx <= x1 + pad and y0 - pad <= my <= y1 + pad

    def frac_at(self, mx):
        x0, _, x1, _ = self.rect
        return max(0.0, min(1.0, (mx - x0) / (x1 - x0)))

    def show_hover(self, mx, my):
        on = self.contains(mx, my) or self.dragging
        self.hover_line.enabled = on
        if on:
            x0, y0, x1, y1 = self.rect
            x = x0 + (x1 - x0) * self.frac_at(mx)
            _place(self.hover_line, x - 0.0012, y0, x + 0.0012, y1)
        return on


def attempts_image(attempts, width=900, height=28):
    """Balken pro Versuch: Höhe = Fortschritt, Farbe = Ergebnis. Viele Versuche teilen sich eine Spalte
    (dann zählt der beste)."""
    img = np.zeros((height, width, 4), np.uint8)
    n = len(attempts)
    if n == 0:
        return Image.fromarray(img)
    prog = np.array([a.fortschritt for a in attempts]) / 100.0
    rank = np.array([3 if a.gueltig and a.auswertung else 2 if a.gueltig else 1 if a.ergebnis == "Crash" else 0
                     for a in attempts])
    cols = np.array([C_ABBRUCH, C_CRASH, C_ZIEL, C_ZIEL_AUSWERTUNG], np.uint8)
    if n <= width:
        bins = np.minimum((np.arange(width) * n) // width, n - 1)      # jede Spalte genau ein Versuch
        best_prog, best_rank = prog[bins], rank[bins]
    else:
        col_of = (np.arange(n) * width) // n
        best_prog = np.zeros(width)
        best_rank = np.zeros(width, int)
        np.maximum.at(best_prog, col_of, prog)
        np.maximum.at(best_rank, col_of, rank)
    heights = np.clip(np.round(best_prog * height), 1, height).astype(int)
    rows = np.arange(height)[:, None]
    mask = rows >= height - heights[None, :]
    img[..., :3] = cols[best_rank][None, :, :]
    img[..., 3] = np.where(mask, 255, 0)
    if n <= width and n < width // 3:            # wenige Versuche: feine Lücken zwischen den Balken
        edges = np.r_[True, bins[1:] != bins[:-1]]
        img[:, edges, 3] = 0
    return Image.fromarray(img)


def speed_image(speed, width=700, height=18, vmax=110.0, total=None):
    """Tempo-Verlauf einer Fahrt: Höhe und Farbe = Tempo (blau langsam, gelb/rot schnell).
    total = Länge der ganzen Leiste in Schritten (fährt das Netz noch, bleibt rechts Platz frei)."""
    img = np.zeros((height, width, 4), np.uint8)
    if len(speed) == 0:
        return Image.fromarray(img)
    total = max(total or len(speed), len(speed))
    idx = (np.arange(width) * total) // width
    v = np.clip(np.abs(speed[np.minimum(idx, len(speed) - 1)]) / vmax, 0, 1)
    v[idx >= len(speed)] = 0.0
    r = np.clip(60 + 400 * v, 0, 255)
    g = np.clip(120 + 200 * v - 250 * np.maximum(v - 0.7, 0) * 3, 40, 230)
    b = np.clip(255 - 330 * v, 30, 255)
    heights = np.clip(np.round(v * height), 1, height).astype(int)
    rows = np.arange(height)[:, None]
    img[..., 0], img[..., 1], img[..., 2] = r[None], g[None], b[None]
    img[..., 3] = np.where((rows >= height - heights[None, :]) & (idx < len(speed))[None, :], 255, 0)
    return Image.fromarray(img)


class InfoBar:
    def __init__(self):
        self.bg = _quad(camera.ui, BAR, z=0.02)
        self.left = TextQuad("", height=0.022, origin=(-.5, 0), px=44, heavy=True)
        self.right = TextQuad("", height=0.022, origin=(.5, 0), px=44, heavy=True)

    def layout(self, A, top_h):
        _place(self.bg, -A / 2, 0.5 - top_h, A / 2, 0.5)
        self.left.position = Vec3(-A / 2 + 0.015, 0.5 - top_h / 2, 0)
        self.right.position = Vec3(A / 2 - 0.015, 0.5 - top_h / 2, 0)

    def set(self, left, right, left_rgb=WHITE):
        self.left.set(left, left_rgb)
        self.right.text = right


class Panel:
    """Rechter Bereich. Kommt in Schritt 8 und 9 (Dashboard); jetzt nur ein paar Zahlen zum Lauf."""

    def __init__(self):
        self.root = Entity(parent=camera.ui)
        self.bg = _quad(self.root, BAR, z=0.02)
        self.title = TextQuad("Dashboard", parent=self.root, height=0.03, origin=(-.5, .5), heavy=True,
                              color=YELLOW)
        self.note = TextQuad("Grafiken kommen in Schritt 8 und 9", parent=self.root, height=0.018,
                             origin=(-.5, .5), px=40, color=GREY)
        self.lines = TextLines(height=0.021, px=42)
        self.lines.parent = self.root

    def layout(self, x0, y0, x1, y1):
        _place(self.bg, x0, y0, x1, y1)
        self.title.position = Vec3(x0 + 0.02, y1 - 0.02, 0)
        self.note.position = Vec3(x0 + 0.02, y1 - 0.06, 0)
        self.lines.position = Vec3(x0 + 0.02, y1 - 0.11, 0)

    def set(self, text):
        self.lines.text = text

    @property
    def enabled(self):
        return self.root.enabled

    @enabled.setter
    def enabled(self, v):
        self.root.enabled = v


class Help:
    def __init__(self, rows):
        self.root = Entity(parent=camera.ui, z=-0.5)
        self.bg = _quad(self.root, color.rgba(0.03, 0.04, 0.1, 0.95), z=0.01)
        self.title = TextQuad("Tastenkürzel", parent=self.root, height=0.034, origin=(-.5, .5), heavy=True,
                              color=YELLOW)
        self.rows = [(TextQuad(k, parent=self.root, height=0.024, origin=(.5, 0), px=44, heavy=True, color=YELLOW),
                      TextQuad(t, parent=self.root, height=0.024, origin=(-.5, 0), px=44)) for k, t in rows]
        self.close = TextQuad("? oder Esc = schliessen", parent=self.root, height=0.019, origin=(0, 0), px=40,
                              color=GREY)
        self.n = len(rows)
        self.root.enabled = False

    ROW = 0.036

    def layout(self, cx, cy):
        h = 0.12 + self.n * self.ROW
        w = 0.8
        _place(self.bg, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        top = cy + h / 2
        self.title.position = Vec3(cx - w / 2 + 0.03, top - 0.025, 0)
        for i, (k, t) in enumerate(self.rows):        # jede Zeile einzeln -> beide Spalten bleiben auf einer Höhe
            y = top - 0.09 - i * self.ROW
            k.position = Vec3(cx - w / 2 + 0.25, y, 0)
            t.position = Vec3(cx - w / 2 + 0.28, y, 0)
        self.close.position = Vec3(cx, cy - h / 2 + 0.022, 0)

    def toggle(self):
        self.root.enabled = not self.root.enabled

    @property
    def visible(self):
        return self.root.enabled
