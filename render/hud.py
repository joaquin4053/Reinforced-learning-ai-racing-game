"""Text-Anzeigen fürs HUD: Text mit Umriss, Stoppuhr mit festen Ziffernbreiten, mehrzeilige Infos.

(Unverändert aus Spiel v1 übernommen.)
"""
import math

from PIL import Image, ImageDraw, ImageFont
from panda3d.core import SamplerState
from ursina import Entity, Texture, Vec3, camera, color
from ursina.shaders import unlit_shader

from game.sim import format_delta, format_time  # noqa: F401 (für play.py)

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
    """Anzeige: Infos oben links, grosse Meldung in der Mitte, Leiste unten:
    links Checkpoints, Mitte Bestzeit | Zeit | Abstand, rechts Tempo."""
    WHITE, YELLOW, GREEN, RED = (255, 255, 255), (255, 215, 40), (90, 255, 120), (255, 90, 80)
    PANEL = color.rgba(0.07, 0.09, 0.2, 0.78)

    def __init__(self, label=None):
        y = -0.445
        self._panel(-0.7, y, 0.18, 0.065)                           # Checkpoints
        for x, w in ((-0.26, 0.24), (0.0, 0.26), (0.26, 0.24)):     # Bestzeit, Zeit, Abstand
            self._panel(x, y, w, 0.065)
        self._panel(0.7, y, 0.22, 0.065)                            # Tempo
        for x, text in ((-0.26, "Bestzeit"), (0.0, "Zeit"), (0.26, "Abstand")):
            TextQuad(text, position=(x, y + 0.058), height=0.024, origin=(0, -.5), heavy=True)
        self.checkpoints = TextQuad(position=(-0.7, y), height=0.04, origin=(0, 0), heavy=True)
        self.best = TextQuad(position=(-0.26, y), height=0.036, origin=(0, 0), heavy=True)
        self.time = MonoText(position=(0, y + 0.024), height=0.045)
        self.delta = TextQuad(position=(0.26, y), height=0.036, origin=(0, 0), heavy=True)
        self.speed = TextQuad(position=(0.7, y), height=0.042, origin=(0, 0), heavy=True)
        self.drift = TextQuad(position=(0.7, y + 0.075), height=0.035, origin=(0, -.5), heavy=True,
                              color=(255, 140, 40))
        self.split = TextQuad(position=(0, 0.2), height=0.045, origin=(0, .5), heavy=True)
        # Zwischenzeiten beim Checkpoint, über den Feldern (kurz eingeblendet)
        self.split_best = TextQuad(position=(-0.26, y + 0.1), height=0.028, origin=(0, -.5), heavy=True)
        self.split_cur = TextQuad(position=(0.0, y + 0.1), height=0.028, origin=(0, -.5), heavy=True)
        self.split_delta = TextQuad(position=(0.26, y + 0.1), height=0.028, origin=(0, -.5), heavy=True)
        self.split_speed = TextQuad(position=(0.7, y + 0.1), height=0.028, origin=(0, -.5), heavy=True)
        self.info = TextLines(position=(-0.86, 0.47), height=0.03)
        self.big = TextQuad(position=(0, 0.02), height=0.09, origin=(0, 0), heavy=True,
                            color=self.YELLOW, align="center")
        self.label = TextQuad(label or "", position=(-0.86, -0.36), height=0.045, origin=(-.5, -.5),
                              heavy=True, color=self.YELLOW)

    @staticmethod
    def _panel(x, y, w, h):
        e = Entity(parent=camera.ui, model="quad", color=HUD.PANEL, shader=unlit_shader,
                   position=Vec3(x, y, 0.01), scale=(w, h))
        e.rotation_z = 0
        return e

    def set(self, speed_ms, t, info="", big=""):
        self.speed.text = f"{abs(speed_ms) * 3.6:3.0f} km/h"
        self.time.text = format_time(t)
        self.info.text = info
        self.big.text = big

    def set_splits(self, best="", cur="", delta="", delta_rgb=None, speed="", speed_rgb=None):
        """Zwischenzeiten über der Leiste (leere Texte = ausblenden)."""
        self.split_best.text = best
        self.split_cur.text = cur
        self.split_delta.set(delta, delta_rgb or self.WHITE)
        self.split_speed.set(speed, speed_rgb or self.WHITE)

    def set_bar(self, checkpoints="", best="–", delta="–", delta_rgb=None):
        self.checkpoints.text = checkpoints
        self.best.text = best
        self.delta.set(delta, delta_rgb or self.WHITE)
