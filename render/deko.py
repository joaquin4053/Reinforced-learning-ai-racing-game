"""Strecken-eigene Kulisse aus der Strecken-Datei ("deko": [...]) – nur Anzeige, keine Kollision, keine Physik.

Jeder Eintrag ist ein Dict mit "art" und einer Stelle entlang der Strecke ("bei" oder "von"/"bis", Meter nach
der Startlinie). "seite": -1 = links, +1 = rechts, 0 = beide/mitte. Alles selbst gezeichnet (eigene Gestaltung).

  {"art": "wand",     "von": 0, "bis": 70, "seite": 0, "hoehe": 14, "farbe": "blau", "abstand": 1}
  {"art": "haus",     "bei": 100, "quer": -30, "breite": 40, "tiefe": 30, "hoehe": 14, "pfeile": "rechts"}
  {"art": "tafeln",   "bei": 320, "muster": "vvXXX", "hoehe": 8}
  {"art": "bruecke",  "bei": 450, "laenge": 18, "hoehe": 12, "pfeile": true}
  {"art": "pfeiler",  "bei": 1370, "laenge": 120, "hoehe": 14}          Hochstrasse quer darüber, mit Stützen
  {"art": "pfeilwand","bei": 3080, "seite": 1, "laenge": 30, "richtung": "links"}
"""
import math

import numpy as np
from PIL import Image, ImageDraw
from ursina import Entity, Texture, Vec3, color

FARBEN = {"blau": (34, 40, 120), "rot": (96, 30, 30), "dunkel": (34, 36, 44), "grau": (70, 74, 84)}
YELLOW, RED_X = (245, 190, 40), (220, 40, 34)
_TEX = {}


def _tex(kind):
    """Schilder: Pfeile nach links/rechts/unten (gelb auf schwarz) oder rotes X."""
    if kind in _TEX:
        return _TEX[kind]
    w, h = (512, 96) if kind in ("links", "rechts") else (128, 128)
    img = Image.new("RGB", (w, h), (22, 22, 26))
    d = ImageDraw.Draw(img)
    if kind in ("links", "rechts"):
        for x in range(10, w, 52):
            s = 1 if kind == "rechts" else -1
            tip, back = (x + 34, x) if s > 0 else (x, x + 34)
            d.polygon([(back, 10), (back + 14 * s, 10), (tip + 14 * s, h // 2), (back + 14 * s, h - 10),
                       (back, h - 10), (tip, h // 2)], fill=YELLOW)
    elif kind == "v":
        for y in (12, 52):
            d.polygon([(14, y), (36, y), (64, y + 34), (92, y), (114, y), (64, y + 60)], fill=YELLOW)
    else:   # X
        d.line([(20, 20), (108, 108)], fill=RED_X, width=22)
        d.line([(108, 20), (20, 108)], fill=RED_X, width=22)
    _TEX[kind] = Texture(img)
    return _TEX[kind]


def _frame(track, dist):
    """(x, y, z, heading, Breite) der Strecke an dieser Stelle."""
    p, e = track.point_at(dist)
    i, _ = track.segment_at(dist)
    y = float(track.center_y[i])
    return float(p[0]), y, float(p[1]), math.atan2(float(e[0]), float(e[1])), float(track.center_width[i])


def _at(x, z, h, quer):
    """Punkt `quer` m rechts der Mittellinie."""
    return x + math.cos(h) * quer, z - math.sin(h) * quer


def _col(name):
    return color.rgb32(*FARBEN.get(name, FARBEN["dunkel"]))


def wand(track, d):
    """Hohe Wand(e) neben der Strecke, folgt ihr in Stücken von 8 m."""
    seiten = [-1, 1] if d.get("seite", 0) == 0 else [d["seite"]]
    hoehe, abstand, col = d.get("hoehe", 14), d.get("abstand", 1.0), _col(d.get("farbe", "dunkel"))
    for s in np.arange(d["von"], d["bis"], 8.0):
        x, y, z, h, w = _frame(track, s + 4)
        for side in seiten:
            cx, cz = _at(x, z, h, side * (w / 2 + abstand + 1.5))
            Entity(model="cube", color=col, position=(cx, y - 20 + (hoehe + 20) / 2, cz),
                   scale=(3, hoehe + 20, 8.4), rotation_y=math.degrees(h))


def haus(track, d):
    """Roter Bau mit Satteldach und Pfeilschild an der Front (z.B. an einer Gabelung)."""
    x, y, z, h, w = _frame(track, d["bei"])
    cx, cz = _at(x, z, h, d.get("quer", 0))
    root = Entity(position=(cx, y, cz), rotation_y=math.degrees(h))
    bw, bt, bh = d.get("breite", 40), d.get("tiefe", 30), d.get("hoehe", 10)
    red = color.rgb32(*FARBEN["rot"])
    Entity(parent=root, model="cube", color=red, position=(0, bh / 2 - 10, 0), scale=(bw, bh + 20, bt))
    for side in (-1, 1):                                    # Dach aus zwei schrägen Platten
        Entity(parent=root, model="cube", color=color.rgb32(70, 22, 22), position=(side * bw / 4, bh + bw * 0.12, 0),
               scale=(bw / 2 * 1.08, 1.0, bt), rotation_z=side * -25)
    pf = d.get("pfeile", "beide")
    for side, kind in ((-1, "links"), (1, "rechts")):
        if pf in ("beide", kind):
            Entity(parent=root, model="quad", texture=_tex(kind), double_sided=True,
                   position=(side * bw / 4, 3.5, -bt / 2 - 0.3), scale=(bw / 2 - 2, 3.2))


def tafeln(track, d):
    """Reihe Schilder quer über der Fahrbahn: v = Durchfahrt, X = gesperrt."""
    x, y, z, h, w = _frame(track, d["bei"])
    muster = d.get("muster", "vX")
    n = len(muster)
    total = d.get("breite", w + 10)
    root = Entity(position=(x, y, z), rotation_y=math.degrees(h))
    hoehe = d.get("hoehe", 8)
    Entity(parent=root, model="cube", color=_col("dunkel"), position=(0, hoehe + total / n / 2 + 1, 0.6),
           scale=(total + 4, total / n + 4, 1.0))
    for k, ch in enumerate(muster):
        px = -total / 2 + total / n * (k + 0.5)
        Entity(parent=root, model="quad", texture=_tex("v" if ch.lower() == "v" else "x"), double_sided=True,
               position=(px, hoehe + total / n / 2 + 1, 0), scale=(total / n - 0.6, total / n - 0.6))


def bruecke(track, d):
    """Dunkle Platte quer über der Strecke (die Strecke führt darunter durch)."""
    x, y, z, h, w = _frame(track, d["bei"])
    root = Entity(position=(x, y, z), rotation_y=math.degrees(h))
    breite, hoehe, laenge = d.get("breite", w + 40), d.get("hoehe", 12), d.get("laenge", 16)
    Entity(parent=root, model="cube", color=_col("dunkel"), position=(0, hoehe + 1.5, 0), scale=(breite, 3, laenge))
    if d.get("pfeile"):
        Entity(parent=root, model="quad", texture=_tex(d.get("richtung", "links")), double_sided=True,
               position=(0, hoehe + 1.5, -laenge / 2 - 0.2), scale=(breite * 0.8, 2.6))


def pfeiler(track, d):
    """Hochstrasse quer über der Strecke mit Stützen links und rechts."""
    x, y, z, h, w = _frame(track, d["bei"])
    root = Entity(position=(x, y, z), rotation_y=math.degrees(h) + d.get("drehung", 90))
    laenge, hoehe = d.get("laenge", 120), d.get("hoehe", 14)
    Entity(parent=root, model="cube", color=_col("grau"), position=(0, hoehe + 1, 0), scale=(laenge, 2.5, 16))
    for px in np.arange(-laenge / 2 + 8, laenge / 2, 16):
        if abs(px) < w / 2 + 3:                             # nicht mitten auf der Fahrbahn
            continue
        for pz in (-6, 6):
            Entity(parent=root, model="cube", color=_col("dunkel"), position=(px, hoehe / 2 - 10, pz),
                   scale=(1.6, hoehe + 20, 1.6))


def pfeilwand(track, d):
    """Pfeilschild am Rand: zeigt die Kurvenrichtung."""
    x, y, z, h, w = _frame(track, d["bei"])
    side = d.get("seite", 1)
    cx, cz = _at(x, z, h, side * (w / 2 + 3))
    laenge = d.get("laenge", 30)
    root = Entity(position=(cx, y, cz), rotation_y=math.degrees(h) + 90)
    Entity(parent=root, model="cube", color=_col("dunkel"), position=(0, 2.5, 0.4), scale=(laenge, 6, 0.6))
    Entity(parent=root, model="quad", texture=_tex(d.get("richtung", "links")), double_sided=True,
           position=(0, 3.5, 0), scale=(laenge - 1, 3.5))


ARTEN = {"wand": wand, "haus": haus, "tafeln": tafeln, "bruecke": bruecke, "pfeiler": pfeiler, "pfeilwand": pfeilwand}


def build(track):
    for d in getattr(track, "deko", []):
        fn = ARTEN.get(d.get("art"))
        if fn is None:
            raise ValueError(f"{track.id}: unbekannte Kulisse '{d.get('art')}'")
        fn(track, d)
