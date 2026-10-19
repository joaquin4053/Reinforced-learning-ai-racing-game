"""Dashboard rechts neben der 3D-Ansicht (Plan-Schritt 8) für viewer.py und watch.py.

Karten untereinander, jede einzeln ein-/ausklappbar (Klick auf den Titel) und mit "?" für eine kurze
Erklärung. Passt nicht alles hinein: Mausrad scrollt. Alles bezieht sich auf den gewählten Versuch und die
aktuelle Stelle der Zeitleiste (rl.playback.Player).

Kurven werden als Bild gezeichnet (PIL) und nur neu gebaut, wenn neue Daten da sind. Was jedes Bild mitläuft
(Markierung, Sensor-Fächer, Aktions-Balken), sind einfache Flächen – das kostet kaum Zeit.
"""
import math
from time import perf_counter

import numpy as np
from panda3d.core import Point3
from PIL import Image, ImageDraw
from ursina import Entity, Vec3, camera, color
from ursina.shaders import unlit_shader

from game.sensors import RAY_ANGLES, RAY_RANGE
from game.sim import format_time
from render.hud import TextLines, TextQuad, _font
from render.viewer_ui import BAR, GREY, WHITE, YELLOW, _place, _quad, _texture
from rl.playback import read_json
from rl.stats import moving_average, official_best, reward_parts, share_valid, training_seconds

HEAD = 0.036                 # Höhe einer Kopfzeile
GAP = 0.008
PX = 1100                    # Pixel pro UI-Einheit für die Kurvenbilder (scharf auch auf Retina)
C_GOOD, C_BAD, C_ABORT = (70, 220, 110), (215, 70, 60), (140, 140, 150)
C_GOLD, C_MINE, C_LINE = (255, 205, 40), (90, 200, 255), (235, 238, 245)
STAGES = {1: "Ankommen", 2: "Zuverlässig", 3: "Schnell (Plateau)", 4: "Schneller als ich"}


def nr(x):
    return f"{x:,}".replace(",", "'")


def _hms(s):
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


# ---------- Kurvenbilder ----------
class Plot:
    """Hilfe zum Zeichnen: Datenbereich -> Pixel. Ränder links für Beschriftung, unten für die x-Achse."""
    L, R, T, B = 74, 10, 8, 26

    def __init__(self, w, h, x0, x1, y0, y1):
        self.img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)
        self.w, self.h = w, h
        self.x0, self.x1 = x0, x1 if x1 > x0 else x0 + 1
        pad = (y1 - y0) * 0.06 or 1.0
        self.y0, self.y1 = y0 - pad, y1 + pad
        self.f = _font(20, False)
        self.d.rectangle([self.L, self.T, w - self.R, h - self.B], fill=(10, 12, 26, 255))

    def px(self, x, y):
        X = self.L + (x - self.x0) / (self.x1 - self.x0) * (self.w - self.L - self.R)
        Y = self.h - self.B - (y - self.y0) / (self.y1 - self.y0) * (self.h - self.T - self.B)
        return X, Y

    def frac_x(self, x):
        """x im Datenbereich -> Anteil an der Bildbreite (für die Markierung)."""
        return self.px(x, self.y0)[0] / self.w

    def data_x(self, frac):
        """Anteil an der Bildbreite -> x im Datenbereich (für Klicks)."""
        return self.x0 + (frac * self.w - self.L) / (self.w - self.L - self.R) * (self.x1 - self.x0)

    def y_labels(self, values, fmt):
        for v in values:
            _, Y = self.px(self.x0, v)
            self.d.line([self.L, Y, self.w - self.R, Y], fill=(255, 255, 255, 28), width=1)
            t = fmt(v)
            self.d.text((self.L - 6 - self.d.textlength(t, font=self.f), Y - 11), t, font=self.f, fill=(190, 195, 210))

    def x_label(self, left, right):
        y = self.h - self.B + 3
        self.d.text((self.L, y), left, font=self.f, fill=(170, 175, 190))
        self.d.text((self.w - self.R - self.d.textlength(right, font=self.f), y), right, font=self.f, fill=(170, 175, 190))


def _ticks(lo, hi, n=3):
    if hi <= lo:
        return [lo]
    step = 10 ** math.floor(math.log10((hi - lo) / n))
    for m in (1, 2, 5, 10):
        if (hi - lo) / (step * m) <= n + 1:
            step *= m
            break
    first = math.ceil(lo / step) * step
    return [first + i * step for i in range(int((hi - first) // step) + 1)]


def learn_image(attempts, w, h, window=50):
    """Reward pro Versuch (Punkte, Farbe = Ergebnis) und gleitender Durchschnitt (Linie)."""
    n = len(attempts)
    r = np.array([a.reward for a in attempts]) if n else np.zeros(0)
    lo, hi = (float(r.min()), float(r.max())) if n else (0.0, 1.0)
    p = Plot(w, h, 0, max(n - 1, 1), lo, hi)
    p.y_labels(_ticks(lo, hi), lambda v: nr(int(round(v))))
    for i, a in enumerate(attempts):
        X, Y = p.px(i, a.reward)
        col = C_GOLD if a.gueltig and a.auswertung else C_GOOD if a.gueltig else C_BAD if a.ergebnis == "Crash" else C_ABORT
        p.d.rectangle([X - 1, Y - 1, X + 1, Y + 1], fill=(*col, 150))
    if n > 1:
        avg = moving_average(r, window)
        p.d.line([p.px(i, v) for i, v in enumerate(avg)], fill=(*C_LINE, 255), width=3)
    p.x_label("Versuch 1", f"Versuch {nr(attempts[-1].versuch)}" if n else "")
    return p


def laptime_image(attempts, best_steps, mine, w, h):
    """Gültige Runden als Punkte (Training hell, Auswertung gold), offizielle Bestzeit als Treppe, deine als Linie."""
    n = len(attempts)
    times = [a.rundenzeit for a in attempts if a.gueltig and a.rundenzeit]
    if mine:
        times.append(mine["time"])
    if not times:
        p = Plot(w, h, 0, max(n - 1, 1), 0, 1)
        p.d.text((p.L + 12, h / 2 - 12), "noch keine gültige Runde", font=p.f, fill=(190, 195, 210))
        return p
    lo, hi = min(times), max(times)
    hi = min(hi, lo + max(20.0, (np.percentile(times, 95) - lo) * 1.3))     # Ausreisser nicht das Bild sprengen lassen
    p = Plot(w, h, 0, max(n - 1, 1), lo, hi)
    p.y_labels(_ticks(lo, hi), lambda v: f"{v:.0f} s")
    index = {a.versuch: i for i, a in enumerate(attempts)}
    for i, a in enumerate(attempts):
        if a.gueltig and a.rundenzeit and a.rundenzeit <= p.y1:
            X, Y = p.px(i, a.rundenzeit)
            if a.auswertung:
                p.d.ellipse([X - 4, Y - 4, X + 4, Y + 4], fill=(*C_GOLD, 255))
            else:
                p.d.rectangle([X - 1, Y - 1, X + 1, Y + 1], fill=(*C_GOOD, 140))
    if best_steps:                                     # offizielle Bestzeit (nur Auswertungen, R6) als Treppe
        pts = []
        for v, t in best_steps:
            X, Y = p.px(index.get(v, 0), t)
            if pts:
                pts.append((X, pts[-1][1]))
            pts.append((X, Y))
        pts.append((p.px(n - 1, best_steps[-1][1])[0], pts[-1][1]))
        p.d.line(pts, fill=(*C_GOLD, 255), width=3)
    if mine:                                           # deine Bestzeit (gestrichelt)
        _, Y = p.px(0, mine["time"])
        x = p.L
        while x < w - p.R:
            p.d.line([x, Y, min(x + 12, w - p.R), Y], fill=(*C_MINE, 255), width=3)
            x += 20
        p.d.text((p.L + 6, Y - 26), f"Du {format_time(mine['time'])}", font=p.f, fill=(*C_MINE, 255))
    p.x_label("Versuch 1", f"Versuch {nr(attempts[-1].versuch)}")
    return p


REWARD_LINES = (("fortschritt", "Fortschritt", (70, 220, 110)), ("zeit", "Zeitstrafe", (255, 140, 60)),
                ("crash", "Crash", (230, 60, 60)), ("ziel", "Ziel", C_GOLD), ("summe", "Summe", C_LINE))


def reward_image(parts, total_frames, w, h):
    """Aufsummierter Reward über die Fahrt, je Bestandteil eine Linie."""
    cum = {k: np.cumsum(v) for k, v in parts.items()}
    vals = np.concatenate([c for c in cum.values()] + [np.zeros(1)])
    lo, hi = float(vals.min()), float(vals.max())
    p = Plot(w, h, 0, max(total_frames, 1), lo, hi)
    p.y_labels(_ticks(lo, hi), lambda v: nr(int(round(v))))
    n = len(parts["summe"])
    step = max(1, n // 400)
    for key, _, col in REWARD_LINES:
        c = cum[key]
        if n and (key in ("fortschritt", "zeit", "summe") or abs(c[-1]) > 0):
            pts = [p.px(i + 1, c[i]) for i in range(0, n, step)] + [p.px(n, c[-1])]
            p.d.line(pts, fill=(*col, 255), width=3 if key == "summe" else 2)
    p.x_label("Start", f"{format_time(total_frames / 60)}")
    return p


# ---------- Karten ----------
class Card:
    key, title, body_h = "", "", 0.1
    help = ""
    text_every = 0.0         # s: Texte höchstens so oft neu (Text = neues Bild, kostet Zeit)

    def __init__(self, dash):
        self.dash = dash
        self.root = Entity(parent=dash.content)
        self.head = _quad(self.root, color.rgba(0.13, 0.16, 0.3, 1), z=0.01)
        self.label = TextQuad(self.title, parent=self.root, height=0.02, origin=(-.5, 0), heavy=True, px=44)
        self.sign = TextQuad("-", parent=self.root, height=0.022, origin=(0, 0), heavy=True, px=44, color=GREY)
        self.qbg = _quad(self.root, color.rgba(0.25, 0.3, 0.55, 1), z=0.005)
        self.q = TextQuad("?", parent=self.root, height=0.02, origin=(0, 0), heavy=True, px=44)
        self.body = Entity(parent=self.root)
        self.collapsed = False
        self.rect = self.head_rect = self.q_rect = (0, 0, 0, 0)
        self.w = 0.0

    def height(self):
        return HEAD + (0 if self.collapsed else self.body_h + GAP)

    def place(self, x0, ytop, w):
        self.w = w
        _place(self.head, x0, ytop - HEAD, x0 + w, ytop)
        self.head_rect = (x0, ytop - HEAD, x0 + w, ytop)
        self.sign.position = Vec3(x0 + 0.014, ytop - HEAD / 2, 0)
        self.label.position = Vec3(x0 + 0.03, ytop - HEAD / 2, 0)
        qx = x0 + w - 0.022
        _place(self.qbg, qx - 0.014, ytop - HEAD + 0.006, qx + 0.014, ytop - 0.006)
        self.q_rect = (qx - 0.018, ytop - HEAD, qx + 0.018, ytop)
        self.q.position = Vec3(qx, ytop - HEAD / 2, 0)
        self.sign.text = "+" if self.collapsed else "-"
        self.body.enabled = not self.collapsed
        y1 = ytop - HEAD - GAP / 2
        self.rect = (x0, y1 - self.body_h, x0 + w, y1)
        self.body.position = Vec3(x0, y1 - self.body_h, 0)
        if not self.collapsed:
            self.build(w, self.body_h)

    def build(self, w, h):
        """Inhalt neu anordnen (lokal: (0,0) = unten links, Breite w, Höhe h)."""

    def refresh(self, ctx):
        """Neue Daten (ca. 1x pro Sekunde)."""

    def frame(self, ctx):
        """Jedes Bild (nur Billiges)."""

    def texts(self, ctx):
        """20x pro Sekunde."""

    def click(self, fx, fy, ctx):
        """Klick in den Inhalt, fx/fy = 0..1. -> True, wenn behandelt."""
        return False

    def hover(self, fx, fy, ctx):
        return ""


class ChartCard(Card):
    """Karte mit einem Kurvenbild und einer senkrechten Markierung."""
    body_h = 0.17

    def __init__(self, dash):
        super().__init__(dash)
        self.img = _quad(self.body, color.white, z=0.01)
        self.marker = _quad(self.body, color.rgba(1, 0.3, 0.3, 0.95), z=-0.01)
        self.guide = _quad(self.body, color.rgba(1, 1, 1, 0.45), z=-0.005)
        self.guide.enabled = False
        self.marker.enabled = False
        self.legend = TextQuad("", parent=self.body, height=0.016, origin=(-.5, .5), px=36, color=GREY)
        self.plot = None
        self._key = None

    def build(self, w, h):
        self.chart_h = h - 0.024
        _place(self.img, 0, 0, w, self.chart_h)
        self.legend.position = Vec3(0.004, h - 0.002, 0)

    def set_plot(self, plot):
        self.plot = plot
        self.img.texture = _texture(plot.img)

    def set_marker(self, frac):
        self.marker.enabled = frac is not None and self.plot is not None
        if self.marker.enabled:
            x = frac * self.w
            _place(self.marker, x - 0.0018, 0.004, x + 0.0018, self.chart_h - 0.004)

    def show_guide(self, fx):
        self.guide.enabled = fx is not None
        if fx is not None:
            x = fx * self.w
            _place(self.guide, x - 0.001, 0.004, x + 0.001, self.chart_h - 0.004)

    def size_px(self):
        return max(64, int(self.w * PX)), max(48, int(self.chart_h * PX))


class ShownCard(Card):
    key, title, body_h = "gezeigt", "Gezeigt", 0.105
    help = ("Was du siehst: welcher Versuch gerade läuft, bei welchem Trainingsschritt er gefahren wurde,\n"
            "welche Geister mitfahren und mit welchen Versionen (Physik, Regeln, Sensoren, Reward).\n\n"
            "Warum das wichtig ist: Ergebnisse verschiedener Versionen darf man nie direkt vergleichen (R8).")

    def __init__(self, dash):
        super().__init__(dash)
        self.lines = TextLines(height=0.018, px=38)
        self.lines.parent = self.body

    def build(self, w, h):
        self.lines.position = Vec3(0.006, h - 0.012, 0)

    def texts(self, ctx):
        self.lines.text = "\n".join(ctx["shown"][:4])


class LearnCard(ChartCard):
    key, title = "lernkurve", "1  Lernkurve"
    help = ("Was du siehst: jeder Punkt ist ein Versuch (grün = Ziel, gold = Ziel in der Auswertung, rot = Crash,\n"
            "grau = Zeitlimit), seine Höhe der Reward. Die weisse Linie ist der Durchschnitt der letzten 50.\n"
            "Die rote Markierung ist der gezeigte Versuch. Klick in die Kurve springt zu diesem Versuch.\n\n"
            "Warum das wichtig ist: Die KI lernt, indem sie den Reward grösser macht. Steigt die Linie, lernt sie.\n"
            "Steigt der Reward, ohne dass die Runden gültiger oder schneller werden, stimmt etwas nicht (R7).")

    def refresh(self, ctx):
        att = ctx["attempts"]
        key = (len(att), self.size_px())
        if key != self._key and att:
            self._key = key
            self.set_plot(learn_image(att, *self.size_px()))
            self.legend.text = "Reward pro Versuch  ·  Linie = Schnitt der letzten 50  ·  Klick = hin"

    def frame(self, ctx):
        i = ctx["focus_index"]
        self.set_marker(self.plot.frac_x(i) if self.plot and i is not None else None)

    def _index_at(self, fx, ctx):
        n = len(ctx["attempts"])
        if not self.plot or not n:
            return None
        return int(round(max(0, min(n - 1, self.plot.data_x(fx)))))

    def click(self, fx, fy, ctx):
        i = self._index_at(fx, ctx)
        if i is not None:
            ctx["jump"](ctx["attempts"][i].versuch)
        return i is not None

    def hover(self, fx, fy, ctx):
        i = self._index_at(fx, ctx)
        if i is None:
            return ""
        a = ctx["attempts"][i]
        return f"Versuch {nr(a.versuch)}: Reward {nr(int(round(a.reward)))}"


class LapCard(LearnCard):
    key, title = "rundenzeit", "2  Rundenzeit"
    help = ("Was du siehst: Zeiten aller gültigen Runden. Kleine grüne Punkte = Training (mit Zufall),\n"
            "goldene = Auswertung (ohne Zufall). Die goldene Treppe ist die offizielle Bestzeit (nur aus\n"
            "Auswertungen, R6), die blaue gestrichelte Linie deine eigene Bestzeit.\n\n"
            "Warum das wichtig ist: Das ist das eigentliche Ziel – regelkonform möglichst schnell. Fällt die\n"
            "Treppe unter deine Linie, ist die KI schneller als du (Stufe 4).")

    def refresh(self, ctx):
        att = ctx["attempts"]
        key = (len(att), self.size_px(), ctx["mine_time"])
        if key != self._key and att:
            self._key = key
            self.set_plot(laptime_image(att, official_best(att), ctx["mine"], *self.size_px()))
            self.legend.text = "gültige Runden  ·  gold = offizielle Bestzeit  ·  blau = du"

    def hover(self, fx, fy, ctx):
        i = self._index_at(fx, ctx)
        if i is None:
            return ""
        a = ctx["attempts"][i]
        return f"Versuch {nr(a.versuch)}: " + (f"Ziel in {format_time(a.rundenzeit)}" if a.gueltig else a.ende)


class StageCard(Card):
    key, title, body_h = "stufen", "3  Erfolgsstufen", 0.12
    help = ("Was du siehst: die vier Erfolgsstufen aus der CLAUDE.md, und bei welchem Versuch sie erreicht wurden.\n"
            "1 Ankommen: erste gültige Runde. 2 Zuverlässig: 5 Auswertungen in Folge im Ziel.\n"
            "3 Schnell: die Bestzeit wird über lange Zeit kaum noch besser (Plateau). 4: schneller als du.\n\n"
            "Warum das wichtig ist: Daran siehst du, wie weit die KI ist – zuerst überhaupt ankommen,\n"
            "dann sicher ankommen, dann schnell werden.")

    def __init__(self, dash):
        super().__init__(dash)
        self.boxes = [_quad(self.body, color.gray) for _ in range(4)]
        self.rows = [TextQuad("", parent=self.body, height=0.017, origin=(-.5, 0), px=38) for _ in range(4)]

    def build(self, w, h):
        for i, (b, r) in enumerate(zip(self.boxes, self.rows)):
            y = h - 0.016 - i * (h - 0.01) / 4
            _place(b, 0.008, y - 0.009, 0.026, y + 0.009)
            r.position = Vec3(0.036, y, 0)

    def texts(self, ctx):
        st = ctx["records"].get("stufen", {})
        for n in range(1, 5):
            s = st.get(str(n))
            self.boxes[n - 1].color = color.rgb32(*C_GOOD) if s else color.rgba(0.3, 0.32, 0.4, 1)
            text = f"{n} {STAGES[n]}"
            if s:
                text += f"  ·  Versuch {nr(s['versuch'])}" if s.get("versuch") else ""
                text += f"  (Schritt {nr(s['schritte'])})"
            else:
                text += "  ·  noch nicht"
            self.rows[n - 1].set(text, WHITE if s else GREY)


class SensorCard(Card):
    key, title, body_h = "sensoren", "4  Sensoren", 0.15
    text_every = 0.25
    help = ("Was du siehst: die 17 Sensorstrahlen der KI als Fächer von links (-90°) bis rechts (+90°).\n"
            "Länge und Farbe = Abstand zur Streckenkante (rot = nah, grün = frei, höchstens 100 m).\n\n"
            "Warum das wichtig ist: Das ist alles, was die KI von der Strecke sieht (R5) – keine Karte,\n"
            "keine Ideallinie. Aus diesen Abständen plus Tempo und Winkel muss sie lernen zu lenken.")

    def __init__(self, dash):
        super().__init__(dash)
        self.full = [Entity(parent=self.body, model="quad", shader=unlit_shader, origin=(0, -.5),
                            color=color.rgba(1, 1, 1, 0.12), z=0.01) for _ in RAY_ANGLES]   # voller Fächer, 100 m
        self.rays = [Entity(parent=self.body, model="quad", shader=unlit_shader, origin=(0, -.5)) for _ in RAY_ANGLES]
        self._last = None
        self.car = _quad(self.body, color.rgb32(255, 122, 16))
        self.info = TextQuad("", parent=self.body, height=0.016, origin=(-.5, .5), px=36, color=GREY)

    def build(self, w, h):
        self.origin = Vec3(w * 0.36, 0.012, 0)
        self.length = min(h - 0.02, w * 0.34)
        _place(self.car, self.origin.x - 0.006, 0.004, self.origin.x + 0.006, 0.022)
        self.info.position = Vec3(w * 0.73, h - 0.008, 0)
        for full, ray, a in zip(self.full, self.rays, RAY_ANGLES):   # Ort und Richtung sind fest
            for e in (full, ray):
                e.position = self.origin
                e.rotation_z = math.degrees(a)                   # Ursina-UI: + = im Uhrzeigersinn = nach rechts
            full.scale = Vec3(0.0025, self.length, 1)
            ray.scale = Vec3(0.005, 0.002, 1)                    # bis zum ersten Bild: kurz (nicht Startgrösse)
        self._last = None

    def frame(self, ctx):
        obs = ctx["obs"]
        vals = tuple(float(k) for k in obs[:17]) if obs is not None else (0.0,) * 17
        if vals == self._last:
            return
        self._last = vals
        for ray, k in zip(self.rays, vals):                      # pro Bild nur Länge und Farbe
            ray.scale = Vec3(0.005, max(0.002, self.length * k), 1)
            ray.color = color.rgba(min(1, 2 * (1 - k)), min(1, 2 * k), 0.15, 0.95)

    def texts(self, ctx):
        obs = ctx["obs"]
        if obs is None:
            self.info.text = ""
            return
        mid = len(RAY_ANGLES) // 2
        self.info.text = (f"links  {obs[0] * RAY_RANGE:3.0f} m\ngerade {obs[mid] * RAY_RANGE:3.0f} m\n"
                          f"rechts {obs[16] * RAY_RANGE:3.0f} m\n\nTempo {abs(obs[17]) * 396:3.0f} km/h\n"
                          f"Winkel {obs[18] * 180:+4.0f}°\n" + ("in der Luft" if obs[22] > 0.5 else "am Boden"))


class ActionCard(Card):
    key, title, body_h = "aktionen", "5  Aktionen", 0.11
    help = ("Was du siehst: was die KI in diesem Moment tut – Lenkung (-1 links bis +1 rechts), Gas, Bremse\n"
            "und Drift (je 0 bis 1). Höchstens eine Entscheidung pro Physik-Schritt (R4).\n\n"
            "Warum das wichtig ist: Hier siehst du ihren Fahrstil, z.B. ob sie früh bremst, ruhig lenkt\n"
            "oder dauernd hin und her zappelt – und wie sich das über die Versuche ändert.")
    NAMES = ("Lenkung", "Gas", "Bremse", "Drift")
    COLORS = ((90, 200, 255), (70, 220, 110), (230, 70, 60), (255, 150, 40))

    def __init__(self, dash):
        super().__init__(dash)
        self.labels = [TextQuad(n, parent=self.body, height=0.016, origin=(-.5, 0), px=36) for n in self.NAMES]
        self.bgs = [_quad(self.body, color.rgba(0.1, 0.12, 0.25, 1), z=0.01) for _ in self.NAMES]
        self.fills = [_quad(self.body, color.rgb32(*c)) for c in self.COLORS]
        self.mid = _quad(self.body, color.rgba(1, 1, 1, 0.6), z=-0.01)
        self.vals = [TextQuad("", parent=self.body, height=0.016, origin=(.5, 0), px=36) for _ in self.NAMES]

    def build(self, w, h):
        self.bar = (0.1, w - 0.07)
        self.ys = [h - 0.016 - i * (h - 0.012) / 4 for i in range(4)]
        for lab, bg, val, y in zip(self.labels, self.bgs, self.vals, self.ys):
            lab.position = Vec3(0.006, y, 0)
            _place(bg, self.bar[0], y - 0.008, self.bar[1], y + 0.008)
            val.position = Vec3(w - 0.006, y, 0)
        cx = (self.bar[0] + self.bar[1]) / 2
        _place(self.mid, cx - 0.001, self.ys[0] - 0.011, cx + 0.001, self.ys[0] + 0.011)
        for fill, y in zip(self.fills, self.ys):              # bis zum ersten Bild: leer (nicht Startgrösse)
            _place(fill, self.bar[0], y - 0.008, self.bar[0] + 0.0005, y + 0.008)

    def frame(self, ctx):
        act = ctx["action"]
        a = [0.0, 0.0, 0.0, 0.0] if act is None else [float(v) for v in act]
        x0, x1 = self.bar
        cx = (x0 + x1) / 2
        for i, (fill, y) in enumerate(zip(self.fills, self.ys)):
            if i == 0:                                         # Lenkung: von der Mitte aus
                xe = cx + a[0] * (x1 - x0) / 2
                _place(fill, min(cx, xe), y - 0.008, max(cx, xe) + 0.0005, y + 0.008)
            else:
                _place(fill, x0, y - 0.008, x0 + max(0.0005, a[i]) * (x1 - x0), y + 0.008)

    def texts(self, ctx):
        act = ctx["action"]
        for i, val in enumerate(self.vals):
            val.text = "" if act is None else f"{float(act[i]):+.2f}" if i == 0 else f"{float(act[i]):.2f}"


class RewardCard(ChartCard):
    key, title = "reward", "6  Reward dieser Fahrt"
    text_every = 0.25
    help = ("Was du siehst: der Reward des laufenden Versuchs, aufsummiert und nach den vier Bestandteilen\n"
            "aus R7 getrennt: Fortschritt (nur neuer Höchststand zählt), Zeitstrafe pro Schritt, Crash-Strafe\n"
            "und Ziel-Bonus. Weiss = Summe. Darunter die Werte dieses einen Schritts.\n\n"
            "Warum das wichtig ist: Genau diese Zahl versucht die KI grösser zu machen. Du siehst, warum\n"
            "Schnellfahren sich lohnt (weniger Zeitstrafe) und wie teuer ein Crash ist.")

    def __init__(self, dash):
        super().__init__(dash)
        self.now = TextQuad("", parent=self.body, height=0.016, origin=(-.5, .5), px=36)

    def build(self, w, h):
        super().build(w, h)
        self.chart_h = h - 0.046
        _place(self.img, 0, 0, w, self.chart_h)
        self.now.position = Vec3(0.004, h - 0.022, 0)

    def refresh(self, ctx):
        self._redraw(ctx)

    def _redraw(self, ctx):
        ep = ctx["episode"]
        if ep is None:
            return
        frames = ep.frames
        total = ep.frames if ep.done else max(frames, int((ep.expected or 0) * 60))
        key = (id(ep), frames, self.size_px())
        if key == self._key:
            return
        if not ep.done and self._key and self._key[0] == id(ep) and perf_counter() - getattr(self, "_t", 0) < 0.5:
            return                                             # fährt noch: höchstens 2x pro Sekunde neu
        self._key, self._t = key, perf_counter()
        self.parts = reward_parts(ep.reward[:frames], ctx["weights"], ctx["result"])
        self.cum = {k: np.cumsum(v) for k, v in self.parts.items()}
        self.total = total
        self.set_plot(reward_image(self.parts, total, *self.size_px()))
        self.legend.text = "grün Fortschritt · orange Zeit · rot Crash · gold Ziel · weiss Summe"

    def frame(self, ctx):
        self._redraw(ctx)
        ep = ctx["episode"]
        if ep is None or self.plot is None:
            self.set_marker(None)
            return
        self.set_marker(self.plot.frac_x(min(ep.frame_at(ctx["t"]), self.total)))

    def texts(self, ctx):
        ep = ctx["episode"]
        if ep is None or self.plot is None or not len(self.parts["summe"]):
            self.now.text = ""
            return
        i = max(0, min(len(self.parts["summe"]) - 1, int(ep.frame_at(ctx["t"])) - 1))
        p, c = {k: v[i] for k, v in self.parts.items()}, {k: v[i] for k, v in self.cum.items()}
        extra = (f"  Crash {p['crash']:+.0f}" if p["crash"] else "") + (f"  Ziel {p['ziel']:+.0f}" if p["ziel"] else "")
        self.now.text = (f"dieser Schritt: Fortschritt {p['fortschritt']:+.2f}  Zeit {p['zeit']:+.2f}{extra}"
                         f"  =  {p['summe']:+.2f}     bis jetzt: {nr(int(round(c['summe'])))}")


class StatsCard(Card):
    key, title, body_h = "statistik", "7  Statistik", 0.11
    help = ("Was du siehst: wie viele der letzten 100 Versuche gültige Runden waren, die offizielle Bestzeit,\n"
            "wie viele Versuche und Schritte es bisher gab und wie lange das Training schon läuft.\n\n"
            "Warum das wichtig ist: Ein Schritt = 1/60 s Fahrt. Die Fahrzeit zeigt, wie viel Erfahrung die KI\n"
            "schon gesammelt hat, die Trainingsdauer, wie lange dein Mac dafür gerechnet hat.")

    def __init__(self, dash):
        super().__init__(dash)
        self.lines = TextLines(height=0.018, px=38)
        self.lines.parent = self.body

    def build(self, w, h):
        self.lines.position = Vec3(0.006, h - 0.012, 0)

    def texts(self, ctx):
        att = ctx["attempts"]
        pct, n = share_valid(att)
        best = ctx["records"].get("bestzeit")
        steps = att[-1].schritte if att else 0
        self.lines.text = "\n".join([
            f"Gültige Runden (letzte {n}): {pct:.0f} %",
            f"Offizielle Bestzeit: {format_time(best) if best else '–'}" +
            (f"   (du: {format_time(ctx['mine_time'])})" if ctx["mine_time"] else ""),
            f"Versuche: {nr(len(att))}   ·   Schritte: {nr(steps)}",
            f"Fahrerfahrung: {_hms(steps / 60)}   ·   Trainingsdauer: {_hms(ctx['train_s'])}",
        ])


CARDS = (ShownCard, LearnCard, LapCard, StageCard, SensorCard, ActionCard, RewardCard, StatsCard)


class Dashboard:
    """Alle Karten in einem scrollbaren, abgeschnittenen Bereich."""

    def __init__(self):
        self.root = Entity(parent=camera.ui)
        self.bg = _quad(self.root, BAR, z=0.03)
        self.clip = Entity(parent=self.root)
        self.content = Entity(parent=self.clip)
        self.cards = [c(self) for c in CARDS]
        self.hover_text = TextQuad("", parent=self.root, height=0.017, origin=(0, -.5), px=36, heavy=True, z=-0.2)
        self.rect = (0, 0, 1, 1)
        self.scroll = 0.0
        self.help_box = HelpBox()
        self._ctx = None
        self._slow = 0.0

    @property
    def enabled(self):
        return self.root.enabled

    @enabled.setter
    def enabled(self, v):
        self.root.enabled = v
        if not v:
            self.help_box.close()

    def layout(self, x0, y0, x1, y1, help_center=(0, 0)):
        self.rect = (x0, y0, x1, y1)
        self.help_center = help_center
        _place(self.bg, x0, y0, x1, y1)
        self.clip.setScissor(Point3(x0, y0, 0), Point3(x1, y1, 0))
        self._place_cards()

    def _content_height(self):
        return sum(c.height() for c in self.cards) + 0.01

    def _place_cards(self):
        x0, y0, x1, y1 = self.rect
        max_scroll = max(0.0, self._content_height() - (y1 - y0))
        self.scroll = max(0.0, min(self.scroll, max_scroll))
        y = y1 - 0.008 + self.scroll
        for c in self.cards:
            c.place(x0 + 0.01, y, x1 - x0 - 0.02)
            y -= c.height()
        if self._ctx:
            self.refresh(self._ctx)
            self.texts(self._ctx)

    def contains(self, mx, my):
        x0, y0, x1, y1 = self.rect
        return self.enabled and x0 <= mx <= x1 and y0 <= my <= y1

    def wheel(self, up):
        self.scroll += -0.06 if up else 0.06
        self._place_cards()

    # ---------- Daten ----------
    def visible_cards(self):
        """Nur Karten, die man gerade sieht (nicht eingeklappt, nicht weggescrollt), werden aktualisiert."""
        _, y0, _, y1 = self.rect
        return [c for c in self.cards if not c.collapsed and c.rect[1] < y1 and c.rect[3] > y0]

    def refresh(self, ctx):
        self._ctx = ctx
        for c in self.visible_cards():
            c.refresh(ctx)

    def frame(self, ctx, mx, my):
        self._ctx = ctx
        for c in self.visible_cards():
            c.frame(ctx)
        hover = ""
        for c in self.cards:
            if isinstance(c, ChartCard):
                inside = not c.collapsed and _inside(c.rect, mx, my) and self.contains(mx, my)
                fx = (mx - c.rect[0]) / max(c.w, 1e-6) if inside else None
                c.show_guide(fx)
                if inside:
                    hover = c.hover(fx, 0, ctx)
                    if hover:
                        self.hover_text.position = Vec3(mx, c.rect[3] - 0.03, 0)
        self.hover_text.text = hover

    def texts(self, ctx):
        self._ctx = ctx
        now = perf_counter()
        for c in self.visible_cards():
            if now - getattr(c, "_text_t", 0.0) >= c.text_every:
                c._text_t = now
                c.texts(ctx)

    # ---------- Maus ----------
    def click(self, mx, my, ctx):
        """-> True, wenn der Klick im Dashboard war."""
        if self.help_box.visible:
            self.help_box.close()
            return self.contains(mx, my)
        if not self.contains(mx, my):
            return False
        for c in self.cards:
            if _inside(c.q_rect, mx, my):
                self.help_box.open(c.title.lstrip("0123456789 "), c.help, self.help_center)
                return True
            if _inside(c.head_rect, mx, my):
                c.collapsed = not c.collapsed
                self._place_cards()
                return True
            if not c.collapsed and _inside(c.rect, mx, my):
                c.click((mx - c.rect[0]) / max(c.w, 1e-6), (my - c.rect[1]) / max(c.body_h, 1e-6), ctx)
                return True
        return True


def _inside(r, mx, my):
    return r[0] <= mx <= r[2] and r[1] <= my <= r[3]


class HelpBox:
    """Erklärung einer Karte ("?"), mitten in der 3D-Ansicht."""

    def __init__(self):
        self.root = Entity(parent=camera.ui, z=-0.7)
        self.bg = _quad(self.root, color.rgba(0.03, 0.04, 0.1, 0.96), z=0.01)
        self.title = TextQuad("", parent=self.root, height=0.03, origin=(-.5, .5), heavy=True, color=YELLOW)
        self.text = TextQuad("", parent=self.root, height=0.021, origin=(-.5, .5), px=42)
        self.close_hint = TextQuad("Klick oder Esc = schliessen", parent=self.root, height=0.017, origin=(0, 0),
                                   px=36, color=GREY)
        self.root.enabled = False

    @property
    def visible(self):
        return self.root.enabled

    def open(self, title, text, center):
        self.title.text, self.text.text = title, text
        cx, cy = center
        w = max(0.5, self.text.scale_x + 0.06)
        h = self.text.scale_y + 0.12
        _place(self.bg, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        self.title.position = Vec3(cx - w / 2 + 0.03, cy + h / 2 - 0.02, 0)
        self.text.position = Vec3(cx - w / 2 + 0.03, cy + h / 2 - 0.065, 0)
        self.close_hint.position = Vec3(cx, cy - h / 2 + 0.018, 0)
        self.root.enabled = True

    def close(self):
        self.root.enabled = False
