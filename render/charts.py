"""Kurvenbilder für das Dashboard (Plan-Schritt 8): Lernkurve, Rundenzeit, Reward einer Fahrt.

Gezeichnet mit PIL in doppelter Grösse und dann verkleinert (glatte Linien). Die Bilder werden nur neu gebaut,
wenn neue Daten da sind; die Markierung des gezeigten Versuchs liegt als eigene Fläche darüber.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

from game.sim import format_time
from render.hud import _font

SS = 2                                   # so viel grösser wird gezeichnet
BG, FRAME = (12, 15, 30), (34, 40, 66)
TEXT, DIM = (205, 210, 225), (130, 136, 158)
C_GOOD, C_BAD = (70, 220, 110), (225, 80, 70)
C_GOLD, C_MINE, C_MEAN = (255, 200, 40), (90, 200, 255), (120, 210, 255)
C_TREND = (110, 235, 140)


def nr(x):
    return f"{x:,}".replace(",", "'")


def nice_ticks(lo, hi, n=4):
    """Runde Zahlen für die Achse (z.B. 0, 1'000, 2'000)."""
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    first = math.ceil(lo / step) * step
    return [first + i * step for i in range(int((hi - first) / step + 1e-9) + 1)]


class Canvas:
    """Zeichenfläche mit Datenbereich. Ränder: links Beschriftung, oben Legende, unten x-Achse."""

    def __init__(self, w, h, x0, x1, y0, y1, left=70, right=12, top=30, bottom=26):
        self.w, self.h = w, h
        self.img = Image.new("RGB", (w * SS, h * SS), BG)
        self.d = ImageDraw.Draw(self.img, "RGBA")              # "RGBA": halb durchsichtig zeichnen
        self.L, self.R, self.T, self.B = left * SS, (w - right) * SS, top * SS, (h - bottom) * SS
        self.x0, self.x1 = x0, (x1 if x1 > x0 else x0 + 1)
        self.y0, self.y1 = y0, (y1 if y1 > y0 else y0 + 1)
        self.font = _font(19 * SS, False)
        self.bold = _font(19 * SS, True)
        self.d.rounded_rectangle([self.L, self.T, self.R, self.B], radius=6 * SS, outline=FRAME, width=SS)

    # ---------- Umrechnung ----------
    def px(self, x, y):
        X = self.L + (x - self.x0) / (self.x1 - self.x0) * (self.R - self.L)
        Y = self.B - (y - self.y0) / (self.y1 - self.y0) * (self.B - self.T)
        return X, Y

    def frac_x(self, x):
        """Datenwert -> Anteil an der Bildbreite (für die Markierung)."""
        return self.px(x, self.y0)[0] / (self.w * SS)

    def data_x(self, frac):
        """Anteil an der Bildbreite -> Datenwert (für Klicks)."""
        return self.x0 + (frac * self.w * SS - self.L) / (self.R - self.L) * (self.x1 - self.x0)

    # ---------- Achsen, Legende ----------
    def grid_y(self, ticks, fmt):
        for v in ticks:
            if not self.y0 <= v <= self.y1:
                continue
            _, Y = self.px(self.x0, v)
            self.d.line([self.L + SS, Y, self.R - SS, Y], fill=(255, 255, 255, 22), width=SS)
            t = fmt(v)
            self.d.text((self.L - 8 * SS - self.d.textlength(t, font=self.font), Y - 11 * SS), t,
                        font=self.font, fill=DIM)

    def ticks_x(self, ticks, fmt):
        for v in ticks:
            if not self.x0 <= v <= self.x1:
                continue
            X, _ = self.px(v, self.y0)
            self.d.line([X, self.B, X, self.B + 4 * SS], fill=FRAME, width=SS)
            t = fmt(v)
            x = min(max(X - self.d.textlength(t, font=self.font) / 2, self.L), self.R - self.d.textlength(t, font=self.font))
            self.d.text((x, self.B + 5 * SS), t, font=self.font, fill=DIM)

    def legend(self, items):
        """items = [(Text, Farbe, "linie" | "punkt" | "band" | "strich")] oben links, nebeneinander."""
        x, y = self.L, 6 * SS
        for text, col, kind in items:
            cy = y + 11 * SS
            if kind == "punkt":
                self.d.ellipse([x + 4 * SS, cy - 4 * SS, x + 12 * SS, cy + 4 * SS], fill=col)
            elif kind == "band":
                self.d.rectangle([x, cy - 6 * SS, x + 16 * SS, cy + 6 * SS], fill=(*col[:3], 70))
            elif kind == "strich":
                for k in (0, 10):
                    self.d.line([x + k * SS, cy, x + (k + 6) * SS, cy], fill=col, width=3 * SS)
            else:
                self.d.line([x, cy, x + 16 * SS, cy], fill=col, width=4 * SS)
            x += 22 * SS
            self.d.text((x, y), text, font=self.font, fill=TEXT)
            x += self.d.textlength(text, font=self.font) + 16 * SS

    # ---------- Formen ----------
    def line(self, pts, col, width=3, dash=False):
        pts = [self.px(x, y) for x, y in pts]
        if len(pts) < 2:
            return
        if not dash:
            self.d.line(pts, fill=col, width=width * SS, joint="curve")
            return
        (xa, ya), (xb, yb) = pts[0], pts[-1]
        n = int(abs(xb - xa) / (18 * SS)) + 1
        for i in range(n):
            t0, t1 = i / n, min(1.0, i / n + 0.55 / n)
            self.d.line([xa + (xb - xa) * t0, ya + (yb - ya) * t0, xa + (xb - xa) * t1, ya + (yb - ya) * t1],
                        fill=col, width=width * SS)

    def band(self, xs, lo, hi, col, alpha=60):
        pts = [self.px(x, y) for x, y in zip(xs, hi)] + [self.px(x, y) for x, y in zip(xs[::-1], lo[::-1])]
        if len(pts) > 2:
            self.d.polygon(pts, fill=(*col[:3], alpha))

    def dot(self, x, y, r, col):
        X, Y = self.px(x, y)
        self.d.ellipse([X - r * SS, Y - r * SS, X + r * SS, Y + r * SS], fill=col)

    def tag(self, x, y, text, col, right=True):
        """Beschriftung mit dunklem Hintergrund an einem Datenpunkt (z.B. Wert am Ende einer Linie)."""
        X, Y = self.px(x, y)
        tw = self.d.textlength(text, font=self.bold)
        x0 = X - tw - 10 * SS if right else X + 6 * SS
        x0 = min(max(x0, self.L + 2 * SS), self.R - tw - 8 * SS)
        y0 = min(max(Y - 13 * SS, self.T + 2 * SS), self.B - 26 * SS)
        self.d.rounded_rectangle([x0 - 4 * SS, y0, x0 + tw + 4 * SS, y0 + 24 * SS], radius=4 * SS, fill=(*BG, 230))
        self.d.text((x0, y0 + 1 * SS), text, font=self.bold, fill=col)

    def message(self, text):
        self.d.text((self.L + 14 * SS, (self.T + self.B) / 2 - 12 * SS), text, font=self.font, fill=DIM)

    def image(self):
        return self.img.resize((self.w, self.h), Image.LANCZOS)


def _rolling(values, positions, window, fn):
    """fn über die letzten `window` Werte bis zu jeder Position (nur an den Stellen positions)."""
    return np.array([fn(values[max(0, p - window + 1):p + 1]) for p in positions])


def _samples(n, k=320):
    return np.unique(np.linspace(0, n - 1, min(n, k)).round().astype(int)) if n else np.zeros(0, int)


def learn_chart(attempts, w, h):
    """Lernkurve: Band = wo die mittleren 80 % der letzten 100 Versuche liegen, Linie = Durchschnitt der letzten 100,
    goldene Punkte = Auswertungen (ohne Zufall)."""
    n = len(attempts)
    r = np.array([a.reward for a in attempts], dtype=np.float64)
    xs = _samples(n)
    if n:
        lo_b = _rolling(r, xs, 100, lambda v: np.percentile(v, 10))
        hi_b = _rolling(r, xs, 100, lambda v: np.percentile(v, 90))
        mean = _rolling(r, xs, 100, np.mean)
        ymin, ymax = min(lo_b.min(), 0.0), max(hi_b.max(), r.max() if n < 50 else hi_b.max())
    else:
        ymin, ymax = 0.0, 1.0
    pad = (ymax - ymin) * 0.08
    c = Canvas(w, h, 0, max(n - 1, 1), ymin - pad, ymax + pad)
    c.grid_y(nice_ticks(ymin, ymax), lambda v: nr(int(round(v))))
    c.legend([("Schnitt der letzten 100", C_MEAN, "linie"), ("mittlere 80 %", C_MEAN, "band"),
              ("Auswertung", C_GOLD, "punkt")])
    if not n:
        c.message("noch keine Versuche")
        return c
    c.ticks_x([t - 1 for t in nice_ticks(1, attempts[-1].versuch, 5) if t >= 1], lambda v: nr(int(v) + 1))
    c.band(xs, lo_b, hi_b, C_MEAN, 70)
    for i, a in enumerate(attempts):
        if a.auswertung:
            c.dot(i, a.reward, 2.2, (*C_GOLD, 200))
    c.line(list(zip(xs, mean)), C_MEAN, 3)
    c.tag(xs[-1], mean[-1], nr(int(round(mean[-1]))), C_MEAN)
    return c


def lap_chart(attempts, best_steps, mine, w, h):
    """Rundenzeit: blasse Punkte = Training (mit Zufall), grüne Linie = Trend, goldene Punkte = Auswertungen,
    goldene Treppe = offizielle Bestzeit (R6), gestrichelt = deine Bestzeit."""
    n = len(attempts)
    valid = [(i, a) for i, a in enumerate(attempts) if a.gueltig and a.rundenzeit]
    times = np.array([a.rundenzeit for _, a in valid]) if valid else np.zeros(0)
    if not len(times):
        c = Canvas(w, h, 0, max(n - 1, 1), 0, 1)
        c.message("noch keine gültige Runde")
        return c
    lo = min([times.min()] + ([mine["time"]] if mine else []))
    hi = max(np.percentile(times, 90), lo + 5)
    c = Canvas(w, h, 0, max(n - 1, 1), lo - 0.8, hi + 0.8)
    c.grid_y(nice_ticks(lo, hi), lambda v: f"{v:.0f} s")
    c.legend([("Training", C_GOOD, "punkt"), ("Trend", C_TREND, "linie"), ("Auswertung", C_GOLD, "punkt"),
              ("Bestzeit", C_GOLD, "linie"), ("Du", C_MINE, "strich")])
    c.ticks_x([t - 1 for t in nice_ticks(1, attempts[-1].versuch, 5) if t >= 1], lambda v: nr(int(v) + 1))
    for i, a in valid:
        if not a.auswertung and a.rundenzeit <= c.y1:
            c.dot(i, a.rundenzeit, 1.6, (*C_GOOD, 70))
    train = [(i, a.rundenzeit) for i, a in valid if not a.auswertung]
    if len(train) > 5:                                     # Trend: Median der letzten 30 gültigen Runden
        ti, tt = np.array([i for i, _ in train]), np.array([t for _, t in train])
        pos = _samples(len(train), 200)
        med = _rolling(tt, pos, 30, np.median)
        c.line(list(zip(ti[pos], np.minimum(med, c.y1))), C_TREND, 2)
    for i, a in valid:
        if a.auswertung and a.rundenzeit <= c.y1:
            c.dot(i, a.rundenzeit, 2.5, (*C_GOLD, 220))
    if best_steps:
        index = {a.versuch: i for i, a in enumerate(attempts)}
        pts = []
        for v, t in best_steps:
            x = index.get(v, 0)
            if pts:
                pts.append((x, pts[-1][1]))
            pts.append((x, t))
        pts.append((n - 1, best_steps[-1][1]))
        c.line([(x, min(t, c.y1)) for x, t in pts], C_GOLD, 4)
        c.tag(n - 1, best_steps[-1][1], format_time(best_steps[-1][1]), C_GOLD)
    if mine:
        c.line([(0, mine["time"]), (n - 1, mine["time"])], C_MINE, 3, dash=True)
        c.tag(0, mine["time"], f"Du {format_time(mine['time'])}", C_MINE, right=False)
    return c


REWARD_LINES = (("fortschritt", "Fortschritt", (90, 225, 120)), ("zeit", "Zeit", (255, 150, 60)),
                ("crash", "Crash", (235, 70, 70)), ("ziel", "Ziel", C_GOLD), ("summe", "Summe", (240, 242, 250)))


def reward_chart(parts, total_frames, w, h):
    """Aufsummierter Reward über die Fahrt, je Bestandteil von R7 eine Linie."""
    cum = {k: np.cumsum(v) for k, v in parts.items()}
    vals = np.concatenate(list(cum.values()) + [np.zeros(1)])
    lo, hi = float(vals.min()), float(vals.max())
    pad = (hi - lo) * 0.08 or 1.0
    c = Canvas(w, h, 0, max(total_frames, 1), lo - pad, hi + pad)
    c.grid_y(nice_ticks(lo, hi), lambda v: nr(int(round(v))))
    c.legend([(name, col, "linie") for _, name, col in REWARD_LINES])
    c.ticks_x([t * 60 for t in nice_ticks(0, total_frames / 60, 5)], lambda v: f"{v / 60:.0f} s")   # x in Schritten
    n = len(parts["summe"])
    if n:
        pos = _samples(n, 400)
        for key, _, col in REWARD_LINES:
            if key in ("fortschritt", "zeit", "summe") or abs(cum[key][-1]) > 0:
                c.line([(p + 1, cum[key][p]) for p in pos], col, 4 if key == "summe" else 2)
        c.tag(pos[-1] + 1, cum["summe"][-1], nr(int(round(cum["summe"][-1]))), (240, 242, 250))
    return c
