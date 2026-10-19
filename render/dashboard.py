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
from render.charts import lap_chart, learn_chart, reward_chart
from render.hud import TextLines, TextQuad
from render.viewer_ui import BAR, GREY, WHITE, YELLOW, _place, _quad, _texture
from rl.netview import group, group_weights
from rl.stats import official_best, reward_parts, share_valid

HEAD = 0.036                 # Höhe einer Kopfzeile
GAP = 0.008
PX = 1100                    # Pixel pro UI-Einheit für die Kurvenbilder (scharf auch auf Retina)
C_GOOD = (70, 220, 110)
STAGES = {1: "Ankommen", 2: "Zuverlässig", 3: "Schnell (Plateau)", 4: "Schneller als ich"}


def nr(x):
    return f"{x:,}".replace(",", "'")


def _hms(s):
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


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
    body_h = 0.2

    def __init__(self, dash):
        super().__init__(dash)
        self.img = _quad(self.body, color.white, z=0.01)
        self.marker = _quad(self.body, color.rgba(1, 0.3, 0.3, 0.95), z=-0.01)
        self.guide = _quad(self.body, color.rgba(1, 1, 1, 0.45), z=-0.005)
        self.guide.enabled = False
        self.marker.enabled = False
        self.plot = None
        self._key = None

    def build(self, w, h):
        self.chart_h = h
        _place(self.img, 0, 0, w, self.chart_h)

    def set_plot(self, plot):
        self.plot = plot
        self.img.texture = _texture(plot.image())

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
    key, title = "lernkurve", "Lernkurve"
    help = ("Was du siehst: den Reward jedes Versuchs über das ganze Training. Die blaue Linie ist der Durchschnitt\n"
            "der letzten 100 Versuche, das Band zeigt, wo die mittleren 80 % liegen (schmal = gleichmässig).\n"
            "Goldene Punkte = Auswertungen ohne Zufall. Rot markiert = gezeigter Versuch, Klick springt hin.\n\n"
            "Warum das wichtig ist: Die KI lernt, indem sie den Reward grösser macht. Steigt die Linie, lernt sie.\n"
            "Steigt der Reward, ohne dass die Runden gültiger oder schneller werden, stimmt etwas nicht (R7).")

    def refresh(self, ctx):
        att = ctx["attempts"]
        key = (len(att), self.size_px())
        if key != self._key and att:
            self._key = key
            self.set_plot(learn_chart(att, *self.size_px()))

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
    key, title = "rundenzeit", "Rundenzeit"
    help = ("Was du siehst: Zeiten aller gültigen Runden. Blasse grüne Punkte = Training (mit Zufall), die grüne\n"
            "Linie ihr Trend, goldene Punkte = Auswertungen (ohne Zufall). Die goldene Treppe ist die offizielle\n"
            "Bestzeit (nur aus Auswertungen, R6), die blaue gestrichelte Linie deine eigene Bestzeit.\n\n"
            "Warum das wichtig ist: Das ist das eigentliche Ziel – regelkonform möglichst schnell. Fällt die\n"
            "Treppe unter deine Linie, ist die KI schneller als du (Stufe 4).")

    def refresh(self, ctx):
        att = ctx["attempts"]
        key = (len(att), self.size_px(), ctx["mine_time"])
        if key != self._key and att:
            self._key = key
            self.set_plot(lap_chart(att, official_best(att), ctx["mine"], *self.size_px()))

    def hover(self, fx, fy, ctx):
        i = self._index_at(fx, ctx)
        if i is None:
            return ""
        a = ctx["attempts"][i]
        return f"Versuch {nr(a.versuch)}: " + (f"Ziel in {format_time(a.rundenzeit)}" if a.gueltig else a.ende)


class StageCard(Card):
    key, title, body_h = "stufen", "Erfolgsstufen", 0.12
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
    key, title, body_h = "sensoren", "Sensoren – was die KI sieht", 0.15
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
    key, title, body_h = "aktionen", "Aktionen – was die KI tut", 0.115
    SMOOTH = 15                 # Schritte (0.25 s): so lange wird der Befehl für die Anzeige gemittelt
    help = ("Was du siehst: was die KI gerade befiehlt – Lenkung (-1 links bis +1 rechts), Gas, Bremse und\n"
            "Drift (je 0 bis 1), gemittelt über die letzte Viertelsekunde. Der weisse Strich bei der Lenkung\n"
            "zeigt, wo das Lenkrad wirklich steht: Es folgt dem Befehl nur mit begrenztem Tempo (Physik).\n\n"
            "Warum das wichtig ist: Hier siehst du ihren Fahrstil. Die KI darf 60 Mal pro Sekunde neu\n"
            "entscheiden (R4) und zappelt oft wild hin und her – ruhiger zu fahren muss sie erst lernen.")
    NAMES = ("Lenkung", "Gas", "Bremse", "Drift")
    COLORS = ((90, 200, 255), (70, 220, 110), (230, 70, 60), (255, 150, 40))

    def __init__(self, dash):
        super().__init__(dash)
        self.labels = [TextQuad(n, parent=self.body, height=0.016, origin=(-.5, 0), px=36) for n in self.NAMES]
        self.bgs = [_quad(self.body, color.rgba(0.1, 0.12, 0.25, 1), z=0.01) for _ in self.NAMES]
        self.fills = [_quad(self.body, color.rgb32(*c)) for c in self.COLORS]
        self.mid = _quad(self.body, color.rgba(1, 1, 1, 0.35), z=-0.01)
        self.wheel = _quad(self.body, color.white, z=-0.02)          # wo das Lenkrad wirklich steht
        self.vals = [TextQuad("", parent=self.body, height=0.016, origin=(.5, 0), px=36) for _ in self.NAMES]
        self.now = np.zeros(4)
        self.wheel_pos = 0.0

    def build(self, w, h):
        self.bar = (0.1, w - 0.075)
        self.ys = [h - 0.016 - i * (h - 0.014) / 4 for i in range(4)]
        for lab, bg, val, y in zip(self.labels, self.bgs, self.vals, self.ys):
            lab.position = Vec3(0.006, y, 0)
            _place(bg, self.bar[0], y - 0.009, self.bar[1], y + 0.009)
            val.position = Vec3(w - 0.006, y, 0)
        cx = (self.bar[0] + self.bar[1]) / 2
        _place(self.mid, cx - 0.0008, self.ys[0] - 0.009, cx + 0.0008, self.ys[0] + 0.009)
        self._show()

    def frame(self, ctx):
        ep, obs = ctx["episode"], ctx["obs"]
        if ep is None or not ep.frames:
            self.now[:] = 0.0
        else:                                          # Mittel der letzten 0.25 s: ruhig und lesbar
            i = min(int(ep.frame_at(ctx["t"])), ep.frames - 1)
            self.now = ep.action[max(0, i - self.SMOOTH + 1):i + 1].astype(np.float64).mean(axis=0)
        self.wheel_pos = float(obs[20]) if obs is not None else 0.0
        self._show()

    def _show(self):
        x0, x1 = self.bar
        cx = (x0 + x1) / 2
        for i, (fill, y) in enumerate(zip(self.fills, self.ys)):
            if i == 0:                                 # Lenkung: von der Mitte aus
                xe = cx + float(np.clip(self.now[0], -1, 1)) * (x1 - x0) / 2
                _place(fill, min(cx, xe), y - 0.009, max(cx, xe) + 0.0005, y + 0.009)
            else:
                _place(fill, x0, y - 0.009, x0 + max(0.0005, float(np.clip(self.now[i], 0, 1))) * (x1 - x0), y + 0.009)
        xw = cx + float(np.clip(self.wheel_pos, -1, 1)) * (x1 - x0) / 2
        _place(self.wheel, xw - 0.0018, self.ys[0] - 0.012, xw + 0.0018, self.ys[0] + 0.012)

    def texts(self, ctx):
        for i, val in enumerate(self.vals):            # dieselben (gemittelten) Werte wie die Balken
            val.text = f"{self.now[i]:+.2f}" if i == 0 else f"{self.now[i]:.2f}"


INPUT_GROUPS = (("Strahlen links", range(0, 6)), ("Strahlen vorne", range(6, 11)), ("Strahlen rechts", range(11, 17)),
                ("Tempo", [17]), ("Winkel", [18]), ("Rutschen", [19]), ("Lenkrad", [20]), ("Steigung", [21]),
                ("in der Luft", [22]), ("senkrecht", [23]))
OUTPUTS = ("Lenkung", "Gas", "Bremse", "Drift")
HIDDEN_DOTS = 16             # je Schicht so viele Punkte (64 Neuronen = je 4 als ein Punkt)
TOP_LINKS = 26               # so viele der stärksten Verbindungen je Schicht werden gezeichnet


class NetCard(Card):
    key, title, body_h = "netz", "Neuronales Netz – wie die KI denkt", 0.25
    text_every = 0.25
    help = ("Was du siehst: das \"Gehirn\" der KI, stark vereinfacht. Links die 24 Beobachtungen (hier zu 10\n"
            "Gruppen zusammengefasst), in der Mitte zwei Schichten mit je 64 Neuronen (je 4 als ein Punkt),\n"
            "rechts die 4 Aktionen. Hell = gerade aktiv. Linien = die stärksten Verbindungen: grün verstärkt,\n"
            "rot hemmt. Das Netz stammt aus dem Snapshot, der dem gezeigten Versuch am nächsten liegt.\n\n"
            "Warum das wichtig ist: Lernen heisst, die Stärke dieser Verbindungen so zu verändern, dass mehr\n"
            "Reward herauskommt. Vergleiche einen frühen mit einem späten Versuch: Die Linien sind anders.")

    def __init__(self, dash):
        super().__init__(dash)
        self.lines = _quad(self.body, color.white, z=0.02)
        self.caption = TextQuad("", parent=self.body, height=0.015, origin=(-.5, .5), px=34, color=GREY)
        self.in_labels = [TextQuad(n, parent=self.body, height=0.0135, origin=(.5, 0), px=32, color=GREY)
                          for n, _ in INPUT_GROUPS]
        self.out_labels = [TextQuad("", parent=self.body, height=0.0145, origin=(-.5, 0), px=32) for _ in OUTPUTS]
        dot = lambda: Entity(parent=self.body, model="circle", shader=unlit_shader, color=color.gray, z=-0.01)  # noqa: E731
        self.nodes = [[dot() for _ in INPUT_GROUPS], [dot() for _ in range(HIDDEN_DOTS)],
                      [dot() for _ in range(HIDDEN_DOTS)], [dot() for _ in OUTPUTS]]
        self.actor = None
        self._lines_key = None
        self._last = None
        self.peak = [1.0, 1.0]                         # grösste Aktivierung je Schicht (für die Helligkeit)
        self.out_now = np.zeros(4)

    def build(self, w, h):
        self.cols = (w * 0.26, w * 0.45, w * 0.62, w * 0.76)
        top, bottom = h - 0.03, 0.01
        self.pos = []
        for col, nodes in zip(self.cols, self.nodes):
            n = len(nodes)
            ys = [top - (top - bottom) * (k + 0.5) / n for k in range(n)]
            self.pos.append([(col, y) for y in ys])
            r = min(0.011, (top - bottom) / n * 0.8)
            for node, y in zip(nodes, ys):
                node.position = Vec3(col, y, -0.01)
                node.scale = Vec3(r, r, 1)
        for lab, (x, y) in zip(self.in_labels, self.pos[0]):
            lab.position = Vec3(x - 0.012, y, 0)
        for lab, (x, y) in zip(self.out_labels, self.pos[3]):
            lab.position = Vec3(x + 0.014, y, 0)
        _place(self.lines, 0, 0, w, h - 0.022)
        self.caption.position = Vec3(0.004, h - 0.002, 0)
        self._lines_key = None
        if self.actor is not None:
            self._draw_lines()

    def refresh(self, ctx):
        actor = ctx["get_actor"]() if ctx.get("get_actor") else None
        if actor is not self.actor:
            self.actor = actor
            self._lines_key = None
            self._last = None
            self.peak = [1.0, 1.0]
        if self.actor is not None and self._lines_key != (id(self.actor), self.w):
            self._draw_lines()
        self.caption.text = (f"{self.actor.label}  ·  24 Eingänge, 2 x 64 Neuronen, 4 Aktionen  ·  dieser Moment"
                             if self.actor else
                             "kein Snapshot gefunden (runs/<name>/models/)")

    def _draw_lines(self):
        """Die stärksten Verbindungen als Bild (ändert sich nur mit dem Netz)."""
        a = self.actor
        W, H = max(64, int(self.w * 1100)), max(48, int((self.body_h - 0.022) * 1100))
        img = Image.new("RGBA", (W * 2, H * 2), (0, 0, 0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        h_img = self.body_h - 0.022

        def P(x, y):
            return x / self.w * W * 2, (1 - y / h_img) * H * 2

        w1 = np.array([[a.w1[r][:, list(idx)].sum() for _, idx in INPUT_GROUPS]
                       for r in np.array_split(np.arange(a.w1.shape[0]), HIDDEN_DOTS)])
        layers = [w1, group_weights(a.w2, HIDDEN_DOTS, HIDDEN_DOTS), group_weights(a.w3, len(OUTPUTS), HIDDEN_DOTS)]
        for li, g in enumerate(layers):
            flat = np.abs(g).ravel()
            order = np.argsort(flat)[::-1][:TOP_LINKS]
            top = flat[order[0]] or 1.0
            for k in order[::-1]:                      # schwächste zuerst, stärkste obenauf
                o, i = divmod(int(k), g.shape[1])
                strength = flat[k] / top
                col = (80, 225, 120) if g[o, i] > 0 else (235, 85, 75)
                (x0, y0), (x1, y1) = self.pos[li][i], self.pos[li + 1][o]
                d.line([P(x0, y0), P(x1, y1)], fill=(*col, int(40 + 180 * strength)), width=max(1, int(1 + 4 * strength)))
        self.lines.texture = _texture(img.resize((W, H), Image.LANCZOS))
        self._lines_key = (id(a), self.w)

    def frame(self, ctx):
        obs = ctx["obs"]
        if self.actor is None or obs is None:
            return
        key = tuple(np.round(obs, 3))
        if key == self._last:
            return
        self._last = key
        h1, h2, act = self.actor.forward(obs)
        self.out_now = act
        ins = [float(np.mean([obs[i] for i in idx])) for _, idx in INPUT_GROUPS]
        for node, v in zip(self.nodes[0], ins):                     # Eingänge: blau = positiv, orange = negativ
            k = min(1.0, abs(v))
            node.color = color.rgba(0.2 + 0.25 * k, 0.25 + 0.6 * k, 0.35 + 0.65 * k, 1) if v >= 0 else \
                color.rgba(0.35 + 0.65 * k, 0.25 + 0.4 * k, 0.15, 1)
        for li, h in ((0, h1), (1, h2)):                              # Neuronen: hell = aktiv
            g = group(h, HIDDEN_DOTS)
            self.peak[li] = max(self.peak[li] * 0.995, float(g.max()), 1e-6)
            for node, v in zip(self.nodes[li + 1], g / self.peak[li]):
                k = float(np.clip(v, 0, 1))
                node.color = color.rgba(0.18 + 0.82 * k, 0.2 + 0.75 * k, 0.32 + 0.35 * k, 1)
        for i, node in enumerate(self.nodes[3]):                       # Aktionen
            k = abs(float(act[i])) if i == 0 else float(act[i])
            node.color = color.rgba(0.2 + 0.8 * k, 0.22 + 0.7 * k, 0.3 + 0.2 * k, 1)

    def texts(self, ctx):
        for i, (lab, name) in enumerate(zip(self.out_labels, OUTPUTS)):
            v = float(self.out_now[i])
            lab.text = f"{name} {v:+.2f}" if i == 0 else f"{name} {v:.2f}"


class RewardCard(ChartCard):
    key, title = "reward", "Reward dieser Fahrt"
    text_every = 0.25
    help = ("Was du siehst: der Reward des laufenden Versuchs, aufsummiert und nach den vier Bestandteilen\n"
            "aus R7 getrennt: Fortschritt (nur neuer Höchststand zählt), Zeitstrafe pro Schritt, Crash-Strafe\n"
            "und Ziel-Bonus. Weiss = Summe. Darüber die Werte dieses einen Schritts.\n\n"
            "Warum das wichtig ist: Genau diese Zahl versucht die KI grösser zu machen. Du siehst, warum\n"
            "Schnellfahren sich lohnt (weniger Zeitstrafe) und wie teuer ein Crash ist.")

    def __init__(self, dash):
        super().__init__(dash)
        self.now = TextQuad("", parent=self.body, height=0.016, origin=(-.5, .5), px=36)

    def build(self, w, h):
        super().build(w, h)
        self.chart_h = h - 0.026
        _place(self.img, 0, 0, w, self.chart_h)
        self.now.position = Vec3(0.004, h - 0.002, 0)

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
        self.set_plot(reward_chart(self.parts, total, *self.size_px()))

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
    key, title, body_h = "statistik", "Statistik", 0.11
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


CARDS = (ShownCard, LearnCard, LapCard, StageCard, SensorCard, NetCard, ActionCard, RewardCard, StatsCard)


class Dashboard:
    """Alle Karten in einem scrollbaren, abgeschnittenen Bereich."""

    def __init__(self):
        self.root = Entity(parent=camera.ui)
        self.bg = _quad(self.root, BAR, z=0.03)
        self.clip = Entity(parent=self.root)
        self.content = Entity(parent=self.clip)
        self.cards = [c(self) for c in CARDS]
        self.hover_bg = _quad(self.root, color.rgba(0.02, 0.03, 0.08, 0.92), z=-0.19)
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
                    if hover:                              # direkt über dem Mauszeiger, im Dashboard
                        self.hover_text.text = hover
                        half = self.hover_text.scale_x / 2 + 0.008
                        x = min(max(mx, self.rect[0] + half), self.rect[2] - half)
                        self.hover_text.position = Vec3(x, my + 0.014, -0.2)
                        _place(self.hover_bg, x - half, my + 0.008, x + half, my + 0.014 + self.hover_text.scale_y)
        self.hover_text.text = hover
        self.hover_bg.enabled = bool(hover)

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
