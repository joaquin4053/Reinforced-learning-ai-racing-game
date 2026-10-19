"""Grafik-Setup. Muss vor dem Start von Ursina importiert werden.

macOS kann moderne OpenGL-Shader nur im Core-Profil (wir nehmen 4.1), und dort nur mit GLSL >= 150 und
`#version` in der allerersten Zeile. Ursinas eingebaute Shader nutzen 130/140 und
beginnen teils mit einer Leerzeile -> ohne diesen Patch bleibt das Bild schwarz.
"""
import re

from panda3d.core import loadPrcFileData

loadPrcFileData("", "gl-version 4 1")
loadPrcFileData("", "framebuffer-multisample 1")
loadPrcFileData("", "multisamples 4")

import ursina.entity  # noqa: E402
from panda3d.core import TransparencyAttrib  # noqa: E402
from ursina.shader import Shader  # noqa: E402


class _Transparency:
    """Ursina setzt M_dual (braucht Alpha-Test aus der alten Pipeline) -> im Core-Profil
    wird dann alles deckend gezeichnet. Stattdessen normales Alpha-Blending."""
    M_dual = TransparencyAttrib.M_alpha

    def __getattr__(self, name):
        return getattr(TransparencyAttrib, name)


ursina.entity.TransparencyAttrib = _Transparency()

from ursina.texture import Texture  # noqa: E402

Texture.default_filtering = "mipmap"  # glatte Texturen statt Pixel-Look

_VERSION = re.compile(r"#version\s+1[0-4]0\b")


def _fix_glsl(src):
    if not src or "#version" not in src:
        return src
    return _VERSION.sub("#version 150", src.lstrip())


_original_compile = Shader.compile


def _compile(self, *args, **kwargs):
    if self.language == Shader.GLSL:
        self.vertex, self.fragment, self.geometry = map(_fix_glsl, (self.vertex, self.fragment, self.geometry))
    return _original_compile(self, *args, **kwargs)


Shader.compile = _compile

# Shader, die Ursina schon beim Import kompiliert hat (z.B. text_shader), neu bauen
import gc  # noqa: E402

from ursina.entity import Entity  # noqa: E402

for _s in [o for o in gc.get_objects() if isinstance(o, Shader)]:
    if getattr(_s, "compiled", False):
        _s.compile()
# ...und Entities, die beim Import schon den alten Shader bekommen haben (z.B. `scene`), neu setzen
for _e in [o for o in gc.get_objects() if isinstance(o, Entity)]:
    if isinstance(getattr(_e, "_shader", None), Shader):
        _e.shader = _e._shader


def fix_after_start():
    """Nach `Ursina()` aufrufen: UI-Linse passend zum Seitenverhältnis setzen.

    Ursina macht das sonst erst bei einem Fenster-Event, das es offscreen nie gibt
    -> UI wäre 40x vergrößert.
    Außerdem das (eigentlich unsichtbare) Vollbild-Overlay abschalten: im Core-Profil
    wird es schwarz über alles gezeichnet. Wir brauchen es nicht.
    """
    from ursina import camera, window
    camera.ui_lens.set_film_size(camera._ui_size * .5 * window.aspect_ratio, camera._ui_size * .5)
    camera.overlay.enabled = False
