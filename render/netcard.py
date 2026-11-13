"""Netz-Panel im Dashboard (Plan-Schritt 9): der Actor von SAC (24 -> N -> N -> 4) im klassischen Stil.
Die Netzgrösse N kommt aus dem gezeigten Netz (train_config.toml [sac] netz, z.B. 64 oder 256).

- Vier Spalten: Eingaben, Schicht 1, Schicht 2, Ausgaben. Neuronen als gleich grosse Kreise, je nach Aktivierung
  gefüllt (inaktiv = nur ein heller Ring). Eingaben mit Namen, Ausgaben mit Namen und Balken.
- Alle Verbindungen immer dünn und hellgrau (Struktur). Darüber die gerade aktiven: Beitrag = Gewicht x Aktivierung
  des Neurons, von dem sie kommt. Über der Schwelle (Regler) farbig: grün = verstärkt, orange = hemmt;
  Dicke und Deckkraft = Stärke. Die Beiträge werden über ca. 0.15 s Fahrzeit geglättet -> kein Flackern.
- Umschalter "alle Neuronen" / "nur die 12 aktivsten pro Schicht" und "Veränderung zeigen" (welche Gewichte sich
  seit dem vorigen Snapshot am stärksten verändert haben).
- Darunter der Critic: geschätzter Wert der Situation über die ganze Fahrt, mit Punkt an der aktuellen Stelle.
- Klick auf ein Neuron: auf einer Karte von oben, wo auf der Strecke es am stärksten aktiv war.

Gerechnet wird mit rl/netview.py: die gespeicherten Beobachtungen des Versuchs gehen durch den Snapshot, der bei
diesem Versuch galt (summary.csv). Das Training wird dafür nicht angefasst.
"""
import math
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
from ursina import Entity, Vec2, Vec3, color
from ursina.models.procedural.circle import Circle
from ursina.shaders import unlit_shader

from game.sim import format_time
from render.charts import critic_chart
from render.dashboard import Card
from render.hud import TextQuad, _font
from render.viewer_ui import GREY, _place, _quad, _texture
from rl.netview import INPUT_NAMES, OUTPUT_NAMES

SIZES = (24, 64, 64, 4)        # Startwert; wird an das gezeigte Netz angepasst (self.sizes)
SHOW_MAX = 64                  # "alle": bei grösseren Schichten höchstens so viele (die aktivsten) – sonst unlesbar
HEADERS = ("Eingaben", "Schicht 1", "Schicht 2", "Ausgaben")
FEW = 12                      # "nur die 12 aktivsten pro Schicht"
MAX_LINES = 50                # höchstens so viele aktive Verbindungen je Schicht
SMOOTH_S = 0.15               # s Fahrzeit: so stark werden Beiträge geglättet
C_PLUS, C_MINUS = (80, 225, 120), (255, 150, 50)
C_FILL = (255, 214, 110)      # Füllung der Neuronen (überall gleich)
C_CHANGE = (190, 120, 255)
PXU = 1100                    # Pixel pro UI-Einheit für Bilder


class _Toggle:
    """Kleiner Knopf in der Karte (lokale Koordinaten)."""

    def __init__(self, parent, text):
        self.bg = _quad(parent, color.rgba(0.16, 0.19, 0.34, 1), z=0.01)
        self.label = TextQuad(text, parent=parent, height=0.0145, origin=(0, 0), px=34, heavy=True)
        self.rect = (0, 0, 0, 0)

    def place(self, x0, y0, x1, y1):
        self.rect = (x0, y0, x1, y1)
        _place(self.bg, x0, y0, x1, y1)
        self.label.position = Vec3((x0 + x1) / 2, (y0 + y1) / 2, 0)

    def set(self, text, on):
        self.label.text = text
        self.bg.color = color.rgb32(200, 70, 60) if on else color.rgba(0.16, 0.19, 0.34, 1)

    def hit(self, x, y):
        x0, y0, x1, y1 = self.rect
        return x0 <= x <= x1 and y0 <= y <= y1


class NetCard(Card):
    key, title, body_h = "netz", "Neuronales Netz – wie die KI denkt", 0.68
    text_every = 0.25
    help = ("Der Actor ist das Netz, das fährt: Er bekommt die 24 Beobachtungen und rechnet daraus die 4 Aktionen.\n"
            "Ein Neuron ist ein Rechenknoten: Es zählt seine Eingänge zusammen – jeder mit einem Gewicht\n"
            "multipliziert – und gibt das Ergebnis weiter, wenn es positiv ist. Diese Zahl ist seine Aktivierung\n"
            "(gefüllter Kreis = aktiv, leerer Ring = gerade still).\n\n"
            "Ein Gewicht ist die Stärke einer Verbindung. Beitrag = Gewicht x Aktivierung: grün schiebt den Wert\n"
            "nach oben (verstärkt), orange nach unten (hemmt). Lernen heisst: die Gewichte so verändern, dass mehr\n"
            "Reward herauskommt. \"Veränderung zeigen\" markiert lila, was sich seit dem vorigen Snapshot am meisten\n"
            "verändert hat. Beim Spulen wechselt das Netz zum Snapshot, der bei diesem Versuch galt.\n\n"
            "Der Critic ist ein zweites Netz, das nicht fährt, sondern bewertet: Wie viel Reward erwartet die KI\n"
            "ab dieser Stelle noch? Fällt die Kurve plötzlich, sieht die KI Gefahr (z.B. vor einer Kurve).\n\n"
            "Klick auf ein Neuron zeigt, an welchen Stellen der Strecke es am stärksten aktiv ist.")

    def __init__(self, dash):
        super().__init__(dash)
        b = self.body
        self.show_all, self.show_change, self.thr = False, False, 0.3
        self.btn_mode = _Toggle(b, "")
        self.btn_change = _Toggle(b, "")
        self.sl_bg = _quad(b, color.rgba(0.1, 0.12, 0.25, 1), z=0.01)
        self.sl_fill = _quad(b, color.rgba(0.45, 0.55, 0.9, 1))
        self.sl_knob = _quad(b, color.white, z=-0.01)
        self.sl_label = TextQuad("", parent=b, height=0.0145, origin=(-.5, 0), px=34)
        self.caption = TextQuad("", parent=b, height=0.0135, origin=(-.5, .5), px=32, color=GREY)
        self.headers = [TextQuad(t, parent=b, height=0.0145, origin=(0, 0), px=34, heavy=True) for t in HEADERS]
        self.structure = _quad(b, color.white, z=0.03)
        self.change_img = _quad(b, color.white, z=0.025)
        self.sizes = list(SIZES)
        self.rings, self.fills = [], []
        self._make_pool()
        self.in_labels = [TextQuad(n, parent=b, height=0.0118, origin=(.5, 0), px=30, color=GREY) for n in INPUT_NAMES]
        self.out_labels = [TextQuad("", parent=b, height=0.0135, origin=(-.5, -.5), px=32) for _ in OUTPUT_NAMES]
        self.out_bg = [_quad(b, color.rgba(0.1, 0.12, 0.25, 1), z=0.01) for _ in OUTPUT_NAMES]
        self.out_fill = [_quad(b, color.rgb32(*C_FILL)) for _ in OUTPUT_NAMES]
        self.lines = [Entity(parent=b, model="quad", shader=unlit_shader, origin=(-.5, 0), z=-0.005, enabled=False)
                      for _ in range(MAX_LINES * 3)]
        self.legend = _quad(b, color.white, z=0.01)
        self.crit_faint = _quad(b, color.white, z=0.02)
        self.crit_bright = _quad(b, color.white, z=0.01)
        self.crit_dot = Entity(parent=b, model="circle", shader=unlit_shader, color=color.white, z=-0.01)
        self.crit_value = TextQuad("", parent=b, height=0.0145, origin=(.5, .5), px=34, heavy=True, color=(255, 200, 40))
        self.actor = self.critic = self.prev = None
        self.shown = self._default_shown()
        self.pos = [{} for _ in self.sizes]
        self.slots = {}                       # aktive Verbindung (Schicht, nach, von) -> Linie
        self._reset_state()
        self._ep_key = self._crit_canvas = None
        self.V = self.H = None
        self._few_t = 0.0
        self._struct_key = self._change_key = None

    def _reset_state(self):
        self.s = None                          # geglättete Beiträge je Schicht
        self.node_s = [np.zeros(n) for n in self.sizes]
        self.peak_c = [1e-6, 1e-6, 1e-6]
        self.peak_n = [1.0, 1.0, 1.0, 1.0]
        self.out_s = np.zeros(4)
        self._t = None

    # ---------- Neuronen (Anzahl je nach Netz) ----------
    def _make_pool(self):
        """Ringe und Füllungen für jedes Neuron (je ein eigenes Modell, Ursina teilt keine Meshes)."""
        for e in [x for layer in self.rings + self.fills for x in layer]:
            e.enabled = False
            e.parent = None
        b = self.body
        self.rings = [[Entity(parent=b, model=Circle(20, mode="line", thickness=1.5), shader=unlit_shader,
                              color=color.rgba(0.82, 0.85, 0.95, 0.75), z=-0.02, enabled=False) for _ in range(n)]
                      for n in self.sizes]
        self.fills = [[Entity(parent=b, model="circle", shader=unlit_shader, color=color.rgb32(*C_FILL), z=-0.015,
                              enabled=False) for _ in range(n)] for n in self.sizes]

    def _limit(self, li):
        """Wie viele Neuronen dieser Schicht gezeigt werden."""
        n = self.sizes[li]
        return min(n, SHOW_MAX) if self.show_all else min(n, FEW)

    def _default_shown(self):
        return [list(range(self.sizes[0])), list(range(self._limit(1))), list(range(self._limit(2))),
                list(range(self.sizes[3]))]

    def _set_sizes(self, actor):
        sizes = [actor.w1.shape[1], actor.w1.shape[0], actor.w2.shape[0], actor.w3.shape[0]]
        if sizes == self.sizes:
            return
        self.sizes = sizes
        self._make_pool()
        self.shown = self._default_shown()
        self.pos = [{} for _ in self.sizes]
        self._reset_state()
        self._few_t = 0.0
        if hasattr(self, "W"):
            self._layout_nodes()
            self._update_controls()

    # ---------- Aufbau ----------
    def build(self, w, h):
        self.W, self.Hh = w, h
        y = h - 0.004
        self.btn_mode.place(0, y - 0.026, w * 0.31, y)
        self.btn_change.place(w * 0.325, y - 0.026, w * 0.58, y)
        self.slider = (w * 0.79, w - 0.012, y - 0.013)
        _place(self.sl_bg, self.slider[0], self.slider[2] - 0.004, self.slider[1], self.slider[2] + 0.004)
        self.sl_label.position = Vec3(w * 0.6, self.slider[2], 0)
        self.caption.position = Vec3(0.004, h - 0.036, 0)
        self.net = (0.0, 0.205, w, h - 0.082)          # x0, y0, x1, y1 des Netzes
        self.cols = (w * 0.22, w * 0.43, w * 0.62, w * 0.77)
        for hd, x in zip(self.headers, self.cols):
            hd.position = Vec3(x, h - 0.066, 0)
        _place(self.legend, 0, 0.172, w, 0.198)
        self.crit = (0.0, 0.0, w, 0.165)
        _place(self.crit_faint, *self.crit)
        self.crit_value.position = Vec3(w - 0.006, self.crit[3] - 0.004, 0)
        self._legend_img()
        self._layout_nodes()
        self._ep_key = None
        self._update_controls()

    def _layout_nodes(self):
        """Neuronen gleichmässig verteilen (nur die gezeigten), Ringe/Füllungen setzen."""
        x0, y0, x1, y1 = self.net
        pad = 0.012
        for li, (col, n) in enumerate(zip(self.cols, self.sizes)):
            idx = self.shown[li]
            m = len(idx)
            step = (y1 - y0 - 2 * pad) / max(m, 1)
            r = min(0.0105, step * 0.36)
            self.pos[li] = {}
            for k, i in enumerate(idx):
                y = y1 - pad - step * (k + 0.5)
                self.pos[li][i] = (col, y)
            for i in range(n):
                on = i in self.pos[li]
                ring, fill = self.rings[li][i], self.fills[li][i]
                if ring.enabled != on:
                    ring.enabled = on
                fill.enabled = False
                if on:
                    ring.position = Vec3(col, self.pos[li][i][1], -0.02)
                    ring.scale = Vec3(2 * r, 2 * r, 1)
                    fill.position = Vec3(col, self.pos[li][i][1], -0.015)
            self.r = getattr(self, "r", {})
            self.r[li] = r
        for i, lab in enumerate(self.in_labels):
            lab.position = Vec3(self.cols[0] - 0.014, self.pos[0][i][1], 0)
        bw = self.W - self.cols[3] - 0.03
        for i in range(4):
            x, y = self.pos[3][i]
            self.out_labels[i].position = Vec3(x + 0.016, y + 0.002, 0)
            _place(self.out_bg[i], x + 0.016, y - 0.012, x + 0.016 + bw, y - 0.004)
        for line in self.lines:
            line.enabled = False
        self.slots.clear()
        self._struct_key = None
        self._draw_structure()
        self._change_key = None

    def _P(self, x, y, W, H):
        """Lokale Koordinate im Netz-Bereich -> Pixel im Bild (W x H)."""
        x0, y0, x1, y1 = self.net
        return (x - x0) / (x1 - x0) * W, (1 - (y - y0) / (y1 - y0)) * H

    def _image_size(self):
        x0, y0, x1, y1 = self.net
        return max(64, int((x1 - x0) * PXU)), max(64, int((y1 - y0) * PXU))

    def _draw_structure(self):
        """Alle Verbindungen dünn und hellgrau – die Struktur des Netzes (ändert sich nur mit der Anordnung)."""
        key = (tuple(map(tuple, self.shown)), self.W)
        if key == self._struct_key:
            return
        self._struct_key = key
        W, H = self._image_size()
        img = Image.new("RGBA", (W * 2, H * 2), (0, 0, 0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        alpha = 14 if self.show_all else 26
        for li in range(3):
            for i in self.shown[li]:
                a = self._P(*self.pos[li][i], W * 2, H * 2)
                for o in self.shown[li + 1]:
                    d.line([a, self._P(*self.pos[li + 1][o], W * 2, H * 2)], fill=(200, 205, 225, alpha), width=1)
        self.structure.texture = _texture(img.resize((W, H), Image.LANCZOS))
        _place(self.structure, *self.net)

    def _legend_img(self):
        W, H = max(64, int(self.W * PXU)), max(16, int(0.026 * PXU))
        img = Image.new("RGBA", (W * 2, H * 2), (0, 0, 0, 0))
        d = ImageDraw.Draw(img, "RGBA")
        f = _font(30, False)
        x, cy = 10, H
        inactive = "inaktiv" if self.show_change else "Verbindung gerade inaktiv"
        items = [("verstärkt", C_PLUS, 6), ("hemmt", C_MINUS, 6), (inactive, (200, 205, 225), 2)]
        if self.show_change:
            items.append(("verändert", C_CHANGE, 5))
        for text, col, wd in items:
            d.line([x, cy, x + 44, cy], fill=(*col, 255), width=wd)
            x += 54
            d.text((x, cy - 17), text, font=f, fill=(205, 210, 225))
            x += d.textlength(text, font=f) + 36
        d.ellipse([x, cy - 12, x + 24, cy + 12], fill=(*C_FILL, 255))
        x += 32
        d.text((x, cy - 17), "aktiv", font=f, fill=(205, 210, 225))
        x += d.textlength("aktiv", font=f) + 26
        d.ellipse([x, cy - 12, x + 24, cy + 12], outline=(210, 215, 235, 220), width=3)
        d.text((x + 32, cy - 17), "still", font=f, fill=(205, 210, 225))
        self.legend.texture = _texture(img.resize((W, H), Image.LANCZOS))

    def _update_controls(self):
        n = max(self.sizes[1], self.sizes[2])
        text = (f"Neuronen: alle {n}" if n <= SHOW_MAX else f"Neuronen: {SHOW_MAX} aktivste") if self.show_all \
            else f"Neuronen: {FEW} aktivste"
        self.btn_mode.set(text, False)
        self.btn_change.set("Veränderung: an" if self.show_change else "Veränderung: aus", self.show_change)
        x0, x1, y = self.slider
        x = x0 + (x1 - x0) * self.thr
        _place(self.sl_fill, x0, y - 0.004, x, y + 0.004)
        _place(self.sl_knob, x - 0.003, y - 0.009, x + 0.003, y + 0.009)
        self.sl_label.text = f"Schwelle {self.thr * 100:.0f} %"

    # ---------- Daten ----------
    def refresh(self, ctx):
        nets = ctx["get_nets"]() if ctx.get("get_nets") else (None, None)
        actor, critic = nets
        if actor is not self.actor:
            self.actor, self.critic = actor, critic
            if actor is not None:
                self._set_sizes(actor)
            self._reset_state()
            self._ep_key = None
            self._change_key = None
        self.caption.text = (f"{self.actor.label}  ·  Actor von SAC: 24 Eingaben, 2 x {self.sizes[1]} Neuronen, 4 Aktionen"
                             if self.actor else "kein Snapshot gefunden (runs/<name>/models/)")
        self._ensure_episode(ctx)
        self._draw_change(ctx)

    def _ensure_episode(self, ctx):
        """Critic-Werte und Aktivierungen für die ganze Fahrt (einmal je Versuch und Snapshot)."""
        ep = ctx["episode"]
        if ep is None or self.actor is None or not ep.frames:
            return
        key = (id(ep), id(self.actor), ep.frames if not ep.done else -1)
        if key == self._ep_key:
            return
        if not ep.done and self._ep_key and self._ep_key[:2] == key[:2] and perf_counter() - getattr(self, "_ep_t", 0) < 0.5:
            return                                              # fährt noch (watch.py): höchstens 2x pro Sekunde
        self._ep_key, self._ep_t = key, perf_counter()
        obs = ep.obs[:ep.frames + 1].astype(np.float64)
        h1, h2, sq, act = self.actor.forward_batch(obs)
        self.H = (obs, h1, h2, act)
        self.V = self.critic.value(obs, sq) if self.critic is not None else None
        if self.V is not None:
            W, H = max(64, int(self.W * PXU)), max(48, int(self.crit[3] * PXU))
            total = len(self.V) if ep.done else max(len(self.V), int((ep.expected or 0) * 60))
            faint = critic_chart(self.V, W, H, bright=False, total=total)
            bright = critic_chart(self.V, W, H, bright=True, total=total)
            self.crit_faint.texture = _texture(faint.image())
            self.crit_bright.texture = _texture(bright.image())
            self._crit_canvas = bright

    def _draw_change(self, ctx):
        """'Veränderung zeigen': die Gewichte, die sich seit dem vorigen Snapshot am stärksten verändert haben (lila)."""
        self.change_img.enabled = self.show_change
        if not self.show_change or self.actor is None:
            return
        key = (id(self.actor), tuple(map(tuple, self.shown)), self.W)
        if key == self._change_key:
            return
        self._change_key = key
        prev = ctx["get_prev"]() if ctx.get("get_prev") else None
        W, H = self._image_size()
        img = Image.new("RGBA", (W * 2, H * 2), (0, 0, 0, 0))
        self.change_note = ""
        if prev is not None:
            d = ImageDraw.Draw(img, "RGBA")
            a, p = self.actor, prev
            for li, (wa, wp) in enumerate(((a.w1, p.w1), (a.w2, p.w2), (a.w3, p.w3))):
                outs, ins = self.shown[li + 1], self.shown[li]
                delta = np.abs(wa - wp)[np.ix_(outs, ins)]
                flat = delta.ravel()
                order = np.argsort(flat)[::-1][:60]
                top = flat[order[0]] if len(order) and flat[order[0]] > 0 else 1.0
                for k in order[::-1]:
                    o, i = divmod(int(k), len(ins))
                    s = flat[k] / top
                    d.line([self._P(*self.pos[li][ins[i]], W * 2, H * 2), self._P(*self.pos[li + 1][outs[o]], W * 2, H * 2)],
                           fill=(*C_CHANGE, int(60 + 170 * s)), width=max(1, int(1 + 5 * s)))
            self.change_note = f"lila = verändert seit {prev.label}"
        self.change_img.texture = _texture(img.resize((W, H), Image.LANCZOS))
        _place(self.change_img, *self.net)

    # ---------- jedes Bild ----------
    def frame(self, ctx):
        obs, ep = ctx["obs"], ctx["episode"]
        if self.actor is None or obs is None:
            return
        self._ensure_episode(ctx)
        t = ctx["t"]
        dt = 0.0 if self._t is None else abs(t - self._t)
        jump = self._t is None or dt > 0.6
        self._t = t
        x = np.asarray(obs, dtype=np.float64)
        h1, h2, act = self.actor.forward(x)
        c = self.actor.contributions(x, h1, h2)
        k = 1.0 if jump else 1.0 - math.exp(-dt / SMOOTH_S)    # Glätten über Fahrzeit (pausiert: steht still)
        if self.s is None or jump:
            self.s = [ci.copy() for ci in c]
        else:
            for si, ci in zip(self.s, c):
                si += (ci - si) * k
        nodes = (np.abs(x), h1, h2, np.abs(act) * np.array([1, 1, 1, 1.0]))
        for li, v in enumerate(nodes):
            self.node_s[li] = v.copy() if jump else self.node_s[li] + (v - self.node_s[li]) * k
        self.out_s = act.copy() if jump else self.out_s + (act - self.out_s) * k
        partial = self._limit(1) < self.sizes[1] or self._limit(2) < self.sizes[2]
        if partial and perf_counter() - self._few_t > 0.5:  # die aktivsten (alle 0.5 s neu)
            self._few_t = perf_counter()
            new = [list(range(self.sizes[0])), sorted(np.argsort(self.node_s[1])[::-1][:self._limit(1)].tolist()),
                   sorted(np.argsort(self.node_s[2])[::-1][:self._limit(2)].tolist()), list(range(self.sizes[3]))]
            if new != self.shown:
                self.shown = new
                self._layout_nodes()
                self._change_key = None
                self._draw_change(ctx)
        self._draw_nodes()
        self._draw_lines()
        self._draw_critic(ctx)

    def _draw_nodes(self):
        for li in range(4):
            v = self.node_s[li]
            if li in (1, 2):
                self.peak_n[li] = max(self.peak_n[li] * 0.997, float(v[self.shown[li]].max()) if self.shown[li] else 1, 1e-6)
                v = v / self.peak_n[li]
            r = self.r[li]
            for i in self.shown[li]:
                kk = float(np.clip(v[i], 0, 1))
                fill = self.fills[li][i]
                on = kk > 0.04
                if fill.enabled != on:
                    fill.enabled = on
                if on:
                    d = 2 * r * (0.3 + 0.7 * math.sqrt(kk))
                    fill.scale = Vec3(d, d, 1)
        bw = self.W - self.cols[3] - 0.03
        for i in range(4):
            x, y = self.pos[3][i]
            a = float(self.out_s[i])
            x0 = x + 0.016
            if i == 0:                                            # Lenkung: von der Mitte aus
                cx = x0 + bw / 2
                xe = cx + float(np.clip(a, -1, 1)) * bw / 2
                _place(self.out_fill[i], min(cx, xe), y - 0.012, max(cx, xe) + 0.0005, y - 0.004)
            else:
                _place(self.out_fill[i], x0, y - 0.012, x0 + max(0.0005, float(np.clip(a, 0, 1))) * bw, y - 0.004)

    def _draw_lines(self):
        """Aktive Verbindungen: über der Schwelle farbig, Dicke/Deckkraft = Stärke, weich ein- und ausgeblendet."""
        want = {}
        for li in range(3):
            S = self.s[li]
            outs, ins = self.shown[li + 1], self.shown[li]
            sub = S[np.ix_(outs, ins)]
            mag = np.abs(sub)
            self.peak_c[li] = max(self.peak_c[li] * 0.995, float(mag.max()) if mag.size else 0, 1e-6)
            rel = mag / self.peak_c[li]
            flat = rel.ravel()
            cand = np.flatnonzero(flat > self.thr)
            if len(cand) > MAX_LINES:
                cand = cand[np.argpartition(flat[cand], -MAX_LINES)[-MAX_LINES:]]
            for kk in cand:
                o, i = divmod(int(kk), len(ins))
                want[(li, outs[o], ins[i])] = (float(flat[kk]), float(sub.ravel()[kk]) > 0)
        for key in [k for k in self.slots if k not in want]:     # nicht mehr aktiv: Linie frei geben
            self.slots.pop(key).enabled = False
        free = [ln for ln in self.lines if not ln.enabled]
        for key, (rel, plus) in want.items():
            line = self.slots.get(key)
            if line is None:
                if not free:
                    continue
                line = free.pop()
                li, o, i = key
                (xa, ya), (xb, yb) = self.pos[li][i], self.pos[li + 1][o]
                line.position = Vec3(xa, ya, -0.005)
                line.rotation_z = -math.degrees(math.atan2(yb - ya, xb - xa))   # Ursina-UI: + = im Uhrzeigersinn
                line.length = math.hypot(xb - xa, yb - ya)
                line.enabled = True
                self.slots[key] = line
            fade = min(1.0, (rel - self.thr) / max(1e-6, (1 - self.thr) * 0.5))      # weich ein-/ausblenden
            col = C_PLUS if plus else C_MINUS
            line.color = color.rgba(col[0] / 255, col[1] / 255, col[2] / 255, 0.15 + 0.8 * fade)
            line.scale = Vec3(line.length, 0.0007 + 0.0028 * rel, 1)

    def _draw_critic(self, ctx):
        ep, c = ctx["episode"], self._crit_canvas
        ok = self.V is not None and c is not None and ep is not None
        for e in (self.crit_faint, self.crit_bright, self.crit_dot):
            e.enabled = ok
        if not ok:
            self.crit_value.text = ""
            return
        f = int(min(max(ep.frame_at(ctx["t"]), 0), len(self.V) - 1))
        fx = c.frac_x(f)
        x0, y0, x1, y1 = self.crit
        w = (x1 - x0) * fx
        _place(self.crit_bright, x0, y0, x0 + max(w, 1e-4), y1)            # bis jetzt hell ...
        self.crit_bright.texture_scale = Vec2(max(fx, 1e-4), 1)             # ... der Rest blass dahinter
        fy = c.frac_y(float(self.V[f]))
        self.crit_dot.position = Vec3(x0 + w, y0 + (y1 - y0) * fy, -0.01)
        self.crit_dot.scale = Vec3(0.009, 0.009, 1)
        self._crit_now = float(self.V[f])

    def texts(self, ctx):
        for i, (lab, name) in enumerate(zip(self.out_labels, OUTPUT_NAMES)):
            v = float(self.out_s[i])
            lab.text = f"{name}  {v:+.2f}" if i == 0 else f"{name}  {v:.2f}"
        if self.V is not None and hasattr(self, "_crit_now"):
            self.crit_value.text = f"Wert jetzt: {self._crit_now:,.0f}".replace(",", "'")
        note = getattr(self, "change_note", "") if self.show_change else ""
        if note:
            self.caption.text = note

    # ---------- Maus ----------
    def click_abs(self, mx, my, ctx):
        x, y = mx - self.rect[0], my - self.rect[1]
        if self.btn_mode.hit(x, y):
            self.show_all = not self.show_all
            self.shown = self._default_shown()
            self._few_t = 0.0
            self._layout_nodes()
            self._update_controls()
            return True
        if self.btn_change.hit(x, y):
            self.show_change = not self.show_change
            self._legend_img()
            self._change_key = None
            self._draw_change(ctx)
            self._update_controls()
            return True
        x0, x1, ys = self.slider
        if x0 - 0.01 <= x <= x1 + 0.01 and abs(y - ys) < 0.016:
            self.drag(mx, my)
            return "drag"
        best = None
        for li in range(4):                                   # Klick auf ein Neuron
            for i, (px, py) in self.pos[li].items():
                d = math.hypot(px - x, py - y)
                if d <= max(self.r[li] * 1.6, 0.008) and (best is None or d < best[0]):
                    best = (d, li, i)
        if best:
            self._explore(best[1], best[2], ctx)
            return True
        return False

    def drag(self, mx, my):
        x0, x1, _ = self.slider
        self.thr = float(np.clip((mx - self.rect[0] - x0) / (x1 - x0), 0.05, 0.95))
        self._update_controls()

    def _explore(self, li, i, ctx):
        """Wo auf der Strecke ist dieses Neuron am stärksten aktiv? -> Karte von oben."""
        ep = ctx["episode"]
        if self.H is None or ep is None:
            return
        obs, h1, h2, act = self.H
        layer = (np.abs(obs), h1, h2, np.abs(act))[li]          # nur die Schicht des angeklickten Neurons
        vals = layer[:, i]
        name = (INPUT_NAMES[i] if li == 0 else OUTPUT_NAMES[i] if li == 3 else f"Neuron {i + 1} in Schicht {li}")
        img = track_map(ctx["track"], ep.pos[:ep.frames + 1], vals, 760, 470)
        f = int(np.argmax(vals))
        text = (f"Hell und gross = stark aktiv, dunkel = still. Am stärksten bei {format_time(f / 60)} "
                f"(Wert {vals[f]:.2f}).\nGerechnet für diesen Versuch: seine Beobachtungen durch {self.actor.label}.")
        self.dash.popup.open(f"{name} – wo ist es aktiv?", img, text, self.dash.help_center)


def track_map(track, pos, values, w, h):
    """Strecke von oben, darauf die Fahrt: je stärker der Wert, desto heller und grösser der Punkt."""
    img = Image.new("RGB", (w * 2, h * 2), (12, 15, 30))
    d = ImageDraw.Draw(img, "RGBA")
    pts = np.vstack([track.left_edge, track.right_edge, pos[:, [0, 2]]])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    scale = min((w * 2 - 60) / max(hi[0] - lo[0], 1), (h * 2 - 60) / max(hi[1] - lo[1], 1))
    ox = (w * 2 - (hi[0] - lo[0]) * scale) / 2
    oy = (h * 2 - (hi[1] - lo[1]) * scale) / 2

    def P(x, z):
        return ox + (x - lo[0]) * scale, h * 2 - (oy + (z - lo[1]) * scale)

    road = [P(*p) for p in track.left_edge] + [P(*p) for p in track.right_edge[::-1]]
    d.polygon(road, fill=(70, 76, 100, 255))
    path = [P(p[0], p[2]) for p in pos[::4]]
    if len(path) > 1:
        d.line(path, fill=(150, 160, 190, 120), width=3)
    v = np.asarray(values, dtype=np.float64)
    top = float(np.percentile(v, 99)) or float(v.max()) or 1.0
    order = np.argsort(v)
    for k in order[::3]:                                   # schwache zuerst, starke obenauf
        s = float(np.clip(v[k] / top, 0, 1))
        if s < 0.15:
            continue
        x, y = P(pos[k][0], pos[k][2])
        r = 3 + 11 * s
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, int(120 + 110 * s), int(60 * (1 - s)), int(60 + 190 * s)))
    sx, sy = P(pos[0][0], pos[0][2])
    d.rectangle([sx - 8, sy - 8, sx + 8, sy + 8], fill=(255, 255, 255, 255))
    return img.resize((w, h), Image.LANCZOS)
