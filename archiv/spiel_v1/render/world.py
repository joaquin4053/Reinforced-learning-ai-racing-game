"""Ursina-Szene aus einem Track bauen + Auto-Modelle + Geister + Sensorstrahlen + HUD.

Rechnet KEINE Physik: bekommt nur Zustände (x, z, heading, ...) zum Anzeigen.
Sim-Koordinate (x, z) -> Ursina Vec3(x, y, z); heading -> rotation_y in Grad.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from panda3d.core import SamplerState
from ursina import AmbientLight, DirectionalLight, Entity, Mesh, Sky, Texture, Vec3, camera, color
from ursina.shaders import lit_with_shadows_shader, unlit_shader

from game.sensors import RAY_ANGLES, RAY_RANGE, ray_distances
from game.sim import format_delta, format_time  # noqa: F401 (für play.py)

WALL_HEIGHT = 1.4
WALL_THICK = 0.8
STRIPE = 8.0  # Meter pro Farbstreifen an der Wand
WALL_COLORS = [color.rgb32(30, 110, 230), color.rgb32(235, 235, 240)]  # Trackmania-Blau/Weiß


# ---------- Texturen ----------
def _checker_texture(n=8, px=16):
    a = (np.indices((n * px, n * px)) // px).sum(axis=0) % 2
    img = np.where(a[..., None] == 1, 245, 20).astype(np.uint8).repeat(3, axis=2)
    return Texture(Image.fromarray(img, "RGB"))


def _asphalt_texture(size=256, seed=1):
    rng = np.random.default_rng(seed)
    fine = rng.normal(0, 6, (size, size))
    coarse = np.kron(rng.normal(0, 4, (size // 16, size // 16)), np.ones((16, 16)))
    img = np.clip(72 + fine + coarse, 0, 255).astype(np.uint8)
    img[:, :3] = img[:, -3:] = 225                     # weiße Randlinien (u = 0 und 1)
    return Texture(Image.fromarray(np.stack([img] * 3, axis=2), "RGB"))


# ---------- Strecke als Meshes ----------
def _road_mesh(track):
    """Straßenband zwischen linkem und rechtem Rand. Textur wiederholt sich alle `width` Meter."""
    verts, uvs, tris = [], [], []
    for i, (l, r) in enumerate(zip(track.left_edge, track.right_edge)):
        v = track.cum[i] / track.width
        verts += [Vec3(l[0], 0.01, l[1]), Vec3(r[0], 0.01, r[1])]
        uvs += [(0, v), (1, v)]
        if i > 0:
            a, b, c, d = 2 * i - 2, 2 * i - 1, 2 * i, 2 * i + 1
            tris += [a, c, b, b, c, d]
    return Mesh(vertices=verts, triangles=tris, uvs=uvs, normals=[Vec3(0, 1, 0)] * len(verts))


def _wall_mesh(track):
    """Beide Banden als ein Mesh: Innenseite, Außenseite, Oberkante; Farbstreifen alle STRIPE m."""
    verts, cols, norms, tris = [], [], [], []

    def quad(p0, p1, p2, p3, n, col):
        k = len(verts)
        verts.extend([p0, p1, p2, p3])
        cols.extend([col] * 4)
        norms.extend([n] * 4)
        tris.extend([k, k + 1, k + 2, k, k + 2, k + 3])

    for edge in (track.left_edge, track.right_edge):
        out = edge - track.center
        out /= np.hypot(out[:, 0], out[:, 1])[:, None]
        outer = edge + out * WALL_THICK
        for i in range(len(edge) - 1):
            col = WALL_COLORS[int(track.cum[i] / STRIPE) % 2]
            a0, a1 = edge[i], edge[i + 1]
            b0, b1 = outer[i], outer[i + 1]
            n_in = Vec3(-out[i][0], 0, -out[i][1])
            lo, hi = 0.0, WALL_HEIGHT
            quad(Vec3(a0[0], lo, a0[1]), Vec3(a0[0], hi, a0[1]), Vec3(a1[0], hi, a1[1]), Vec3(a1[0], lo, a1[1]), n_in, col)
            quad(Vec3(b1[0], lo, b1[1]), Vec3(b1[0], hi, b1[1]), Vec3(b0[0], hi, b0[1]), Vec3(b0[0], lo, b0[1]), -n_in, col)
            quad(Vec3(a0[0], hi, a0[1]), Vec3(b0[0], hi, b0[1]), Vec3(b1[0], hi, b1[1]), Vec3(a1[0], hi, a1[1]), Vec3(0, 1, 0), col)
    return Mesh(vertices=verts, triangles=tris, colors=cols, normals=norms)


def _gate(gate, width, banner_texture=None, banner_color=color.white, height=8):
    """Tor über der Strecke: zwei Pfeiler + Banner."""
    t = gate.tangent
    root = Entity(position=Vec3(gate.center[0], 0, gate.center[1]), rotation_y=math.degrees(math.atan2(t[0], t[1])))
    for side in (-1, 1):
        Entity(parent=root, model="cube", color=color.rgb32(30, 30, 35),
               position=(side * (width / 2 + WALL_THICK + 0.6), height / 2, 0), scale=(1.2, height, 1.2))
    Entity(parent=root, model="cube", texture=banner_texture, color=banner_color, position=(0, height, 0),
           scale=(width + 2 * WALL_THICK + 2.4, 1.6, 0.8), texture_scale=(width / 2, 1))
    return root


# ---------- Text (Ursinas eigener Text geht im macOS-Core-Profil nicht) ----------
_FONTS = {}
_FONT_PATHS = {
    True: ["/System/Library/Fonts/Supplemental/Arial Black.ttf", "ariblk.ttf", "DejaVuSans-Bold.ttf"],
    False: ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"],
}


def _font(px, heavy):
    if (px, heavy) not in _FONTS:
        for path in _FONT_PATHS[heavy]:
            try:
                _FONTS[(px, heavy)] = ImageFont.truetype(path, px)
                break
            except OSError:
                continue
        else:
            _FONTS[(px, heavy)] = ImageFont.load_default(px)
    return _FONTS[(px, heavy)]


class TextQuad(Entity):
    """Text als PIL-Bild auf einem Quad – mit Umriss, damit er überall lesbar ist.

    height = Höhe einer Zeile (UI-Einheiten bzw. Meter in der Welt).
    origin wie bei Ursina: (-.5, .5) = Position ist die linke obere Ecke.
    """

    def __init__(self, text="", position=(0, 0, 0), height=0.04, origin=(-.5, .5), color=(255, 255, 255),
                 heavy=False, align="left", px=64, parent=None, **kwargs):
        super().__init__(parent=parent or camera.ui, model="quad", position=Vec3(*position) if len(position) == 3
                         else Vec3(position[0], position[1], 0), origin=origin, shader=unlit_shader, **kwargs)
        self.height, self.heavy, self.align, self.px = height, heavy, align, px
        self._rgb = color
        self._text = None
        self.text = text

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._render(value, self._rgb)

    def set(self, text, rgb=None):
        self._render(text, rgb or self._rgb)

    def _render(self, value, rgb):
        if value == self._text and rgb == self._rgb:
            return
        self._text, self._rgb = value, rgb
        self.visible = bool(value)
        if not value:
            return
        # Schon einmal gezeichnet? Dann die fertige Textur wiederverwenden (spart Zeit -> kein Ruckeln)
        key = (value, rgb, self.px, self.heavy, self.align)
        if key not in _TEXT_CACHE:
            if len(_TEXT_CACHE) > 2000:
                _TEXT_CACHE.clear()
            _TEXT_CACHE[key] = _make_text_texture(value, rgb, self.px, self.heavy, self.align)
        tex, w, h = _TEXT_CACHE[key]
        self.texture = tex
        self.scale = Vec3(w / self.px * self.height, h / self.px * self.height, 1)


_TEXT_CACHE = {}


def _to_texture(img):
    tex = Texture(img)
    tex._texture.setMinfilter(SamplerState.FT_linear)
    tex._texture.setMagfilter(SamplerState.FT_linear)
    return tex


def _make_text_texture(value, rgb, px, heavy, align):
    f = _font(px, heavy)
    stroke, pad = max(2, px // 14), px // 6
    box = ImageDraw.Draw(Image.new("RGBA", (1, 1))).multiline_textbbox(
        (0, 0), value, font=f, stroke_width=stroke, spacing=px // 5, align=align)
    w, h = math.ceil(box[2] - box[0]) + 2 * pad, math.ceil(box[3] - box[1]) + 2 * pad
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).multiline_text(
        (pad - box[0], pad - box[1]), value, font=f, fill=(*rgb, 255), stroke_width=stroke,
        stroke_fill=(0, 0, 0, 220), spacing=px // 5, align=align)
    return _to_texture(img), w, h


class MonoText(Entity):
    """Text aus einzelnen Zeichen-Feldern gleicher Größe (für die Stoppuhr).

    Jedes Zeichen wird nur einmal gezeichnet; danach werden nur noch fertige Texturen getauscht.
    Ziffern stehen dadurch immer an derselben Stelle und nichts ruckelt.
    """

    NARROW = ":. "

    def __init__(self, position=(0, 0), height=0.05, color=(255, 255, 255), heavy=True, px=64):
        super().__init__(parent=camera.ui, position=Vec3(position[0], position[1], 0))
        self.height, self.rgb, self.heavy, self.px = height, color, heavy, px
        f = _font(px, heavy)
        stroke, pad = max(2, px // 14), px // 6
        top, bottom = f.getbbox("0123456789", stroke_width=stroke)[1::2]
        self._cell_h = bottom - top + 2 * pad
        self._cell_w = math.ceil(max(f.getlength(d) for d in "0123456789")) + stroke
        self._top, self._pad, self._stroke = top, pad, stroke
        self._glyphs = {}
        self._cells = []
        self._text = None

    def _glyph(self, ch):
        if ch not in self._glyphs:
            w = int(self._cell_w * 0.45) if ch in self.NARROW else self._cell_w
            img = Image.new("RGBA", (w, self._cell_h), (0, 0, 0, 0))
            f = _font(self.px, self.heavy)
            x = (w - f.getlength(ch)) / 2
            ImageDraw.Draw(img).text((x, self._pad - self._top), ch, font=f, fill=(*self.rgb, 255),
                                     stroke_width=self._stroke, stroke_fill=(0, 0, 0, 220))
            self._glyphs[ch] = (_to_texture(img), w)
        return self._glyphs[ch]

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        if value == self._text:
            return
        self._text = value
        while len(self._cells) < len(value):
            self._cells.append(Entity(parent=self, model="quad", origin=(0, .5), shader=unlit_shader))
        scale = self.height / self.px
        widths = [self._glyph(ch)[1] for ch in value]
        x = -sum(widths) * scale / 2
        for cell, ch, w in zip(self._cells, value, widths):
            cell.enabled = True
            cell.texture = self._glyph(ch)[0]
            cell.scale = Vec3(w * scale, self._cell_h * scale, 1)
            cell.x = x + w * scale / 2
            x += w * scale
        for cell in self._cells[len(value):]:
            cell.enabled = False


# ---------- Autos ----------
class CarModel(Entity):
    """Auto aus Würfeln: Karosserie, Kabine, Spoiler, Räder (Vorderräder lenken mit)."""

    def __init__(self, body_color=color.rgb32(235, 70, 40), alpha=1.0, **kwargs):
        super().__init__(**kwargs)

        def part(col, **kw):
            return Entity(parent=self, model="cube", color=color.rgba(col.r, col.g, col.b, alpha), **kw)

        dark, black = color.rgb32(20, 20, 25), color.rgb32(15, 15, 15)
        part(body_color, scale=(2.0, 0.5, 4.0), y=0.55)
        part(body_color.tint(-.25), scale=(1.9, 0.25, 1.4), y=0.55, z=1.4)
        part(color.rgb32(25, 30, 45), scale=(1.55, 0.45, 1.7), y=1.0, z=-0.3)
        part(dark, scale=(2.1, 0.08, 0.5), y=1.25, z=-1.85)
        for sx in (-0.75, 0.75):
            part(dark, scale=(0.1, 0.4, 0.1), x=sx, y=1.0, z=-1.85)
            part(color.rgb32(255, 250, 210), scale=(0.4, 0.15, 0.05), x=sx, y=0.6, z=2.0, shader=unlit_shader)
            part(color.rgb32(255, 30, 30), scale=(0.4, 0.15, 0.05), x=sx, y=0.6, z=-2.0, shader=unlit_shader)
        self.front_wheels = []
        for wx in (-1.0, 1.0):
            for wz in (-1.3, 1.3):
                w = part(black, scale=(0.35, 0.7, 0.7), position=(wx, 0.35, wz))
                if wz > 0:
                    self.front_wheels.append(w)

    def set_state(self, x, z, heading, steer=0.0):
        self.position = Vec3(x, 0, z)
        self.rotation_y = math.degrees(heading)
        for w in self.front_wheels:
            w.rotation_y = steer * 30


class GhostCar:
    """Halb durchsichtiges Auto eines Geists mit Namensschild. Keine Kollision (nur Anzeige)."""

    COLORS = [color.rgb32(60, 200, 255), color.rgb32(190, 110, 255), color.rgb32(120, 255, 140), color.rgb32(255, 200, 60)]

    def __init__(self, ghost, index=0):
        self.ghost = ghost
        col = self.color = self.COLORS[index % len(self.COLORS)]
        self.model = CarModel(body_color=col, alpha=0.45)
        self.model.hide(0b0001)  # wirft keinen Schatten
        self.anchor = Entity(parent=self.model, y=3.0)  # Punkt über dem Auto für das Schild
        # Schild in Bildschirm-Größe (bleibt lesbar und wird nie riesig)
        self.label = TextQuad(parent=camera.ui, height=0.03, origin=(0, -.5), heavy=True)
        self.reset()

    def reset(self):
        self.crashed = False
        self.label.set(self.ghost.label + ("" if self.ghost.compatible else "  (alte Version)"), (255, 255, 255))

    def show_frame(self, frame):
        """frame darf eine Kommazahl sein -> weich zwischen zwei gespeicherten Schritten."""
        f0 = math.floor(frame)
        x, z, h, _ = lerp_pose((*self.ghost.state_at(f0), 0), (*self.ghost.state_at(f0 + 1), 0), frame - f0)
        self.model.set_state(x, z, h)
        if frame >= self.ghost.last_frame and self.ghost.meta["result"] == "crashed" and not self.crashed:
            self.crashed = True  # bleibt an der Crash-Stelle stehen, mit Hinweis
            self.label.set(f"{self.ghost.label} · Crash {self.ghost.meta['progress_pct']:.0f} %", (255, 120, 100))
        # Schild auf den Bildschirm projizieren; ausblenden, wenn der Geist hinter/zu nah an der Kamera ist
        to_ghost = self.anchor.world_position - camera.world_position
        dist = to_ghost.length()
        in_front = to_ghost.dot(camera.forward) > 0
        self.label.enabled = in_front and dist > 6 and bool(self.label.text)
        if self.label.enabled:
            p = self.anchor.screen_position
            self.label.position = Vec3(p.x, p.y, 0)


class RayViz(Entity):
    """Sensorstrahlen der KI als leuchtende Balken mit Trefferpunkt (rot = Wand nah, grün = frei)."""

    def __init__(self, track, **kwargs):
        super().__init__(**kwargs)
        self.track = track
        self.beams = [Entity(parent=self, model="cube", shader=unlit_shader) for _ in RAY_ANGLES]
        self.dots = [Entity(parent=self, model="cube", shader=unlit_shader, scale=0.45) for _ in RAY_ANGLES]

    def set_distances(self, x, z, heading, distances):
        y = 0.8
        for beam, dot, a, dist in zip(self.beams, self.dots, heading + RAY_ANGLES, distances):
            k = min(1.0, dist / 40.0)
            c = color.rgba(1 - k, 0.25 + 0.75 * k, 0.2, 0.8)
            sx, sz = math.sin(a), math.cos(a)
            beam.position = Vec3(x + sx * dist / 2, y, z + sz * dist / 2)
            beam.rotation_y = math.degrees(a)
            beam.scale = Vec3(0.08, 0.08, dist)
            beam.color = c
            dot.position = Vec3(x + sx * dist, y, z + sz * dist)
            dot.color = c
            dot.visible = dist < RAY_RANGE

    def set_state(self, x, z, heading):
        """Für Replays: Strahlen aus einer Position neu berechnen (gleiche Funktion wie in der Sim)."""
        self.set_distances(x, z, heading, ray_distances(x, z, heading, self.track.walls))


# ---------- Welt ----------
class World:
    def __init__(self, track, shadows=True):
        self.track = track
        lo = track.center.min(axis=0) - 150
        hi = track.center.max(axis=0) + 150
        mid, size = (lo + hi) / 2, hi - lo
        if shadows:
            Entity.default_shader = lit_with_shadows_shader

        Sky(texture="sky_sunset")
        Entity(model="plane", texture="grass", color=color.rgb32(120, 170, 90),
               scale=(size[0], 1, size[1]), position=(mid[0], -0.02, mid[1]),
               texture_scale=(size[0] / 12, size[1] / 12))
        Entity(model=_road_mesh(track), texture=_asphalt_texture(), color=color.white, double_sided=True)
        Entity(model=_wall_mesh(track), double_sided=True)

        checker = _checker_texture()
        start = track.start_gate
        _gate(start, track.width, checker)
        if track.finish_gate is not start and not track.is_loop:
            _gate(track.finish_gate, track.width, checker)
        for g in ([start] if track.is_loop else [start, track.finish_gate]):
            t = g.tangent
            Entity(model="plane", texture=checker, position=(g.center[0], 0.03, g.center[1]),
                   rotation_y=math.degrees(math.atan2(t[0], t[1])), scale=(track.width, 1, 3),
                   texture_scale=(track.width / 3, 1))
        for g in track.checkpoints:
            _gate(g, track.width, banner_color=color.rgb32(255, 200, 40), height=6)

        # Bäume als Deko (fester Seed -> immer gleich)
        rng = np.random.default_rng(7)
        for _ in range(260):
            x, z = rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1])
            _, _, _, lateral = track.progress_at(x, z)
            if abs(lateral) < track.width / 2 + 8:
                continue
            h = rng.uniform(5, 11)
            Entity(model="cube", color=color.rgb32(90, 60, 35), position=(x, h * 0.25, z), scale=(0.7, h * 0.5, 0.7))
            Entity(model="icosphere", color=color.rgb32(40, int(rng.uniform(100, 150)), 50),
                   position=(x, h * 0.7, z), scale=h * 0.55)

        self.sun = DirectionalLight(shadow_map_resolution=(4096, 4096), shadows=shadows)
        self.sun.look_at(Vec3(0.6, -1, 0.4))
        AmbientLight(color=color.rgba(0.55, 0.55, 0.6, 1))
        if shadows:
            bounds = Entity(model="cube", position=(mid[0], 3, mid[1]), scale=(size[0] - 250, 8, size[1] - 250), visible=False)
            self.sun.update_bounds(bounds)

        self.car = CarModel()
        self.rays = RayViz(track)
        self.rays.enabled = False
        camera.clip_plane_far = 3000


class TextLines(Entity):
    """Mehrzeiliger Text, bei dem jede Zeile ein eigenes TextQuad ist.

    Ändert sich nur eine Zeile (z.B. "Checkpoint 2/5"), wird nur diese neu gezeichnet -> kein Ruckeln.
    """

    def __init__(self, position=(0, 0), height=0.03, px=40):
        super().__init__(parent=camera.ui, position=Vec3(position[0], position[1], 0))
        self.height, self.px = height, px
        self.lines = []

    @property
    def text(self):
        return "\n".join(line.text or "" for line in self.lines)

    @text.setter
    def text(self, value):
        rows = value.split("\n") if value else []
        while len(self.lines) < len(rows):
            self.lines.append(TextQuad(parent=self, position=(0, -len(self.lines) * self.height * 1.25, 0),
                                       height=self.height, px=self.px))
        for line, row in zip(self.lines, rows):
            line.text = row
        for line in self.lines[len(rows):]:
            line.text = ""


class HUD:
    WHITE, YELLOW, GREEN, RED = (255, 255, 255), (255, 215, 40), (90, 255, 120), (255, 90, 80)

    def __init__(self, label=None):
        self.speed = TextQuad(position=(0.84, -0.42), height=0.075, origin=(.5, -.5), heavy=True)
        self.drift = TextQuad(position=(0.84, -0.3), height=0.045, origin=(.5, -.5), heavy=True,
                              color=(255, 140, 40))
        self.time = MonoText(position=(0, 0.47), height=0.05)
        self.split = TextQuad(position=(0, 0.2), height=0.045, origin=(0, .5), heavy=True)
        self.info = TextLines(position=(-0.86, 0.47), height=0.03)
        self.big = TextQuad(position=(0, -0.02), height=0.09, origin=(0, 0), heavy=True,
                            color=self.YELLOW, align="center")
        self.label = TextQuad(label or "", position=(-0.86, -0.42), height=0.05, origin=(-.5, -.5),
                              heavy=True, color=self.YELLOW)

    def set(self, speed_ms, t, info="", big=""):
        self.speed.text = f"{abs(speed_ms) * 3.6:3.0f} km/h"
        self.time.text = format_time(t)
        self.info.text = info
        self.big.text = big


def lerp_pose(a, b, t):
    """Zwischen zwei Posen (x, z, heading, steer) überblenden, t = 0..1. Winkel über den kürzeren Weg."""
    dh = (b[2] - a[2] + math.pi) % (2 * math.pi) - math.pi
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + dh * t, a[3] + (b[3] - a[3]) * t)


