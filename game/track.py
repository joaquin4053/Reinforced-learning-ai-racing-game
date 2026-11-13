"""Strecke aus Bausteinen laden (tracks/<map>.json).

Format:
{
  "name": "Einstieg",
  "version": 1,                 bei jeder Änderung erhöhen -> alte Geister/Bestzeiten zählen nicht mehr
  "width": 28,                  Fahrbahnbreite in Metern (inkl. Rand)
  "rand": false,                Streckenrand als Standard für alle Bausteine (einzeln überschreibbar)
  "start": [x, y, z, richtung], Startpunkt und Fahrtrichtung in Grad (0 = +z, 90 = +x)
  "pieces": [                   Bausteine in Fahrtrichtung, jeder setzt am Ende des vorherigen an:
    ["start"],                  Startlinie (das Auto steht kurz dahinter)
    ["straight", 40],           Gerade, Länge in m
    ["turn", 90, 30],           Kurve: Winkel in Grad (+ = rechts, - = links), Radius der Mittellinie
    ["turn", 90, 60, 25],       ... mit Überhöhung (Steilkurve) in Grad, baut sich weich auf und ab
    ["slope", 40, 6],           Steigung/Gefälle: Länge, Höhenänderung (weich ein- und auslaufend)
    ["ramp", 14, 3],            Absprung-Rampe: steigt gleichmässig und endet mit einer Kante
    ["gap", 20, -2],            Lücke ohne Fahrbahn: Länge, Höhe der Landung relativ zur Kante
    ["straight", 200, {"rand": true}],   jeder Baustein: optional zuletzt {"rand": true/false} = Kante am Rand
                                         (Standard aus "rand" der Strecke, sonst ohne)
    ["turn", 90, 80, {"breite": 50}],    ... und/oder {"breite": m}: Breite ändert sich weich über den Baustein
    ["checkpoint"],             Checkpoint (in dieser Reihenfolge zu durchfahren, Regel R1)
    ["finish"]                  Ziel
  ],
  "deko": [...]                 optional: Kulisse nur zur Anzeige (Wände, Häuser, Schilder, Brücken), render/deko.py
}

Daraus entstehen:
- die Mittellinie in 3D (für Fortschritt, Checkpoints, Runterfallen),
- eine Höhenkarte der Fahrbahn im Raster RES (für Physik und Sensoren): Höhe oder NaN (= keine Fahrbahn).
Die Fahrbahn schwebt frei: Wer über die Kante fährt, fällt (Regel R2).
"""
import json
import math
from pathlib import Path

import numpy as np

RES = 0.25          # m Rasterweite der Höhenkarte
SAMPLE = 0.1        # m Abstand der Abtastpunkte entlang der Mittellinie beim Aufbau
CENTER_STEP = 1.0   # m Abstand der Mittellinienpunkte für Fortschritt und Darstellung
START_OFFSET = 6.0  # das Auto steht so viele Meter nach der Startlinie
GATE_EXTRA = 1.0    # Tore sind so viel breiter als die Fahrbahn
BANK_RAMP = 0.25    # Anteil einer Steilkurve, über den sich die Überhöhung auf- bzw. abbaut
KANTE_WIDTH = 0.3   # m: Kante am Rand ("rand": true): niedrige, runde Stufe ganz aussen ...
KANTE_HEIGHT = 0.4  # m: ... etwa halbe Radhöhe. So steil, dass man nicht hochfahren kann: langsam quer
                    #    hinein -> prallt ab, zu schnell quer hinein -> hüpft drüber (config.toml [kollision])
BAND_WIDTH = 1.4    # m: davor ein flacher, aufgemalter rot-weisser Streifen (nur Anzeige)
CURB_WIDTH = BAND_WIDTH + KANTE_WIDTH    # so breit ist der Randbereich (für Tore/Anzeige)


def curb_profile(lat, width, border=False):
    """Zusätzliche Höhe der Fahrbahn quer zur Fahrtrichtung (lat = Abstand zur Mitte, m).

    width und border dürfen Arrays sein (passend zu lat). Mit Rand: ganz aussen eine niedrige, runde
    Stufe (KANTE_HEIGHT). Ohne Rand: flach bis zur Abbruchkante."""
    border = np.asarray(border, dtype=bool)
    t = np.clip((np.abs(lat) - (np.asarray(width) / 2 - KANTE_WIDTH)) / KANTE_WIDTH, 0.0, 1.0)
    rise = np.clip(t / 0.7, 0.0, 1.0)
    rise = rise * rise * (3 - 2 * rise)                 # steil hoch, oben rund
    return np.where(border, KANTE_HEIGHT * rise, 0.0)


class Gate:
    """Eine Linie quer über die Strecke (Start, Checkpoint oder Ziel)."""

    def __init__(self, center, y, tangent, half_width, dist, name):
        self.center = np.asarray(center, dtype=np.float64)   # (x, z)
        self.y = float(y)
        self.tangent = np.asarray(tangent, dtype=np.float64)
        left = np.array([-self.tangent[1], self.tangent[0]])
        self.a = self.center + left * half_width
        self.b = self.center - left * half_width
        self.dist = dist
        self.name = name

    def crossed(self, x0, z0, x1, z1):
        """Hat die Bewegung (x0,z0)->(x1,z1) diese Linie vorwärts überquert?"""
        if (x1 - x0) * self.tangent[0] + (z1 - z0) * self.tangent[1] <= 0:
            return False
        return _segments_intersect((x0, z0), (x1, z1), self.a, self.b)


def _segments_intersect(p1, p2, q1, q2):
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1, d2 = cross(q1, q2, p1), cross(q1, q2, p2)
    d3, d4 = cross(p1, p2, q1), cross(p1, p2, q2)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)


def build_path(start, pieces, border_default=False, width_default=16.0):
    """Bausteine -> dichte Abtastpunkte entlang der Mittellinie.

    Rückgabe: dict mit Arrays x, y, z, heading, road (Fahrbahn ja/nein), kind (Baustein-Index),
    tilt (Querneigung: Höhe = tilt × seitlicher Abstand, + = rechts höher) und
    marks: Liste (Abtastindex, "start"/"checkpoint"/"finish").
    """
    x, y, z, h = float(start[0]), float(start[1]), float(start[2]), math.radians(start[3])
    xs, ys, zs, hs, road, kind, tilt, border, marks = [x], [y], [z], [h], [True], [0], [0.0], [border_default], []
    widths = [float(width_default)]
    rand = border_default
    w_now = w_from = w_to = float(width_default)

    def add(nx, ny, nz, nh, on_road, k, tl=0.0):
        xs.append(nx), ys.append(ny), zs.append(nz), hs.append(nh), road.append(on_road), kind.append(k)
        tilt.append(tl), border.append(rand), widths.append(w_now)

    for k, piece in enumerate(pieces):
        name, args = piece[0], list(piece[1:])
        opts = args.pop() if args and isinstance(args[-1], dict) else {}
        rand = bool(opts.get("rand", border_default))
        w_from, w_to = widths[-1], float(opts.get("breite", widths[-1]))
        if name in ("start", "checkpoint", "finish"):
            marks.append((len(xs) - 1, name))
            continue
        if name == "turn":
            angle, radius = math.radians(args[0]), float(args[1])
            length = abs(angle) * radius
        else:
            length = float(args[0])
        n = max(1, int(round(length / SAMPLE)))
        y0 = y
        for i in range(1, n + 1):
            t = i / n
            on_road = True
            tw = t * t * (3 - 2 * t)                            # Breite weich ändern
            w_now = w_from + (w_to - w_from) * tw
            if name == "straight":
                ny = y0
            elif name == "turn":
                h_new = h + angle / n
                x += math.sin((h + h_new) / 2) * length / n      # Sehne in der Mitte des Teilbogens
                z += math.cos((h + h_new) / 2) * length / n
                h = h_new
                bank = math.radians(float(args[2])) if len(args) > 2 else 0.0
                ramp = min(1.0, t / BANK_RAMP, (1 - t) / BANK_RAMP)
                ramp = ramp * ramp * (3 - 2 * ramp)               # weich ein- und auslaufen
                inner = 1.0 if angle > 0 else -1.0                # Innenseite (rechts bei Rechtskurve) tiefer
                add(x, y0, z, h, True, k, -inner * math.tan(bank) * ramp)
                continue
            elif name == "slope":
                ny = y0 + float(args[1]) * (1 - math.cos(math.pi * t)) / 2
            elif name == "ramp":
                ny = y0 + float(args[1]) * t
            elif name == "gap":
                ny = y0 + float(args[1]) * t
                on_road = False
            else:
                raise ValueError(f"Unbekannter Baustein '{name}' (Nr. {k + 1})")
            x += math.sin(h) * length / n
            z += math.cos(h) * length / n
            add(x, ny, z, h, on_road, k)
        y = ys[-1]
    return {"x": np.array(xs), "y": np.array(ys), "z": np.array(zs), "heading": np.array(hs),
            "road": np.array(road), "kind": np.array(kind), "tilt": np.array(tilt), "border": np.array(border),
            "width": np.array(widths), "marks": marks}


class Track:
    def __init__(self, data, path=None):
        self.path = str(path) if path else None
        self.id = Path(path).stem if path else data.get("name", "map")
        self.name = data.get("name", self.id)
        self.deko = list(data.get("deko", []))       # Kulisse nur zur Anzeige (render/deko.py), keine Physik
        self.version = int(data.get("version", 1))
        self.width = float(data.get("width", 12))
        self.is_loop = False                       # Bausteinstrecken gehen von Start bis Ziel
        self.laps = 1
        self.pieces = data["pieces"]
        p = build_path(data["start"], self.pieces, bool(data.get("rand", False)), self.width)
        self.width = float(p["width"].max())                   # breiteste Stelle (Tore, Anzeige)
        self.samples = p

        # ---- Mittellinie (grob, 1 m) für Fortschritt, Tore, Darstellung ----
        seg = np.hypot(np.diff(p["x"]), np.diff(p["z"]))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        idx = np.unique(np.concatenate([np.searchsorted(cum, np.arange(0, cum[-1], CENTER_STEP)), [len(cum) - 1]]))
        self._sample_cum = cum
        self.center = np.stack([p["x"][idx], p["z"][idx]], axis=1)
        self.center_y = p["y"][idx]
        self.center_road = p["road"][idx]
        seg = np.diff(self.center, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.seg_dir = seg / self.seg_len[:, None]
        tangents = np.stack([np.sin(p["heading"][idx]), np.cos(p["heading"][idx])], axis=1)
        left = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
        self.center_width = p["width"][idx]
        self.center_tilt = p["tilt"][idx]
        self.left_edge = self.center + left * self.center_width[:, None] / 2
        self.right_edge = self.center - left * self.center_width[:, None] / 2

        # ---- Start, Checkpoints, Ziel ----
        marks = p["marks"]
        names = [m[1] for m in marks]
        if names.count("start") != 1 or names.count("finish") != 1:
            raise ValueError(f"{self.id}: braucht genau einen 'start' und ein 'finish'")
        start_i = dict((m[1], m[0]) for m in marks)["start"]
        self._dist_offset = cum[start_i]
        self.checkpoints, finish = [], None
        for si, name in marks:
            g = self._gate(si, p, cum, name)
            if name == "checkpoint":
                g.name = f"CP {len(self.checkpoints) + 1}"
                self.checkpoints.append(g)
            elif name == "finish":
                finish = g
            else:
                self.start_gate = g
        self.finish_gate = finish
        self.length = finish.dist
        h = p["heading"][start_i]
        self.start_heading = float(h)
        sx = p["x"][start_i] + math.sin(h) * START_OFFSET
        sz = p["z"][start_i] + math.cos(h) * START_OFFSET
        self.start_pos = (float(sx), float(sz))
        self._build_heightmap(p)
        self.start_y = self.ground(sx, sz)
        self._validate()

    @staticmethod
    def load(path):
        path = Path(path)
        return Track(json.loads(path.read_text()), path)

    def _gate(self, si, p, cum, name):
        h = p["heading"][si]
        tangent = (math.sin(h), math.cos(h))
        dist = cum[si] - self._dist_offset
        g = Gate((p["x"][si], p["z"][si]), p["y"][si], tangent, p["width"][si] / 2 + GATE_EXTRA, dist, name)
        g.width = float(p["width"][si])
        return g

    # ---------- Höhenkarte ----------
    def _build_heightmap(self, p):
        """Fahrbahn quer zur Mittellinie in ein Raster stempeln: Höhe je Zelle, NaN = keine Fahrbahn."""
        on = p["road"]
        xs, ys, zs, hs, tl = p["x"][on], p["y"][on], p["z"][on], p["heading"][on], p["tilt"][on]
        bd, ws = p["border"][on], p["width"][on]
        si = np.nonzero(on)[0]
        lat = np.arange(-self.width / 2, self.width / 2 + 1e-9, SAMPLE)
        inside = np.abs(lat[None, :]) <= ws[:, None] / 2 + 1e-9   # jede Stelle so breit wie dort vorgesehen
        rx, rz = np.cos(hs)[:, None], -np.sin(hs)[:, None]
        px = (xs[:, None] + rx * lat[None, :])[inside]
        pz = (zs[:, None] + rz * lat[None, :])[inside]
        py = (ys[:, None] + tl[:, None] * lat[None, :]
              + curb_profile(lat[None, :], ws[:, None], bd[:, None]))[inside]
        owner = np.broadcast_to(self._sample_cum[si][:, None], inside.shape)[inside]   # Meter entlang der Strecke
        margin = 2.0
        self.x0, self.z0 = px.min() - margin, pz.min() - margin
        nx = int(math.ceil((px.max() + margin - self.x0) / RES)) + 1
        nz = int(math.ceil((pz.max() + margin - self.z0) / RES)) + 1
        ix = np.floor((px - self.x0) / RES).astype(np.int64)
        iz = np.floor((pz - self.z0) / RES).astype(np.int64)
        flat = iz * nx + ix
        total = np.bincount(flat, weights=py, minlength=nx * nz)       # Mittelwert je Zelle -> glatte Rampen
        count = np.bincount(flat, minlength=nx * nz)
        with np.errstate(invalid="ignore", divide="ignore"):
            height = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        lo = np.full(nx * nz, np.inf)
        hi = np.full(nx * nz, -np.inf)
        np.minimum.at(lo, flat, owner)
        np.maximum.at(hi, flat, owner)
        crowded = (hi - lo) > 3 * self.width
        if crowded.any():
            c = int(np.argmax(crowded))
            raise ValueError(f"{self.id}: Strecke überlappt sich bei ca. ({self.x0 + c % nx * RES:.0f}, "
                             f"{self.z0 + c // nx * RES:.0f}) – Brücken/Kreuzungen gibt es (noch) nicht")
        self.height = height.reshape(nz, nx)
        self.nx, self.nz = nx, nz
        self.min_y = float(np.nanmin(self.height))

    def ground(self, x, z):
        """Höhe der Fahrbahn bei (x, z), oder None wenn dort keine Fahrbahn ist."""
        return self.ground_slope(x, z)[0]

    def ground_slope(self, x, z):
        """(Höhe, Steigung in x, Steigung in z) der Fahrbahn bei (x, z). Höhe None = keine Fahrbahn,
        Steigung None = am Rand (dort ist sie unbekannt)."""
        fx = (x - self.x0) / RES - 0.5
        fz = (z - self.z0) / RES - 0.5
        i, j = math.floor(fx), math.floor(fz)
        if i < 0 or j < 0 or i + 1 >= self.nx or j + 1 >= self.nz:
            return None, None, None
        h = self.height
        h00, h10, h01, h11 = h.item(j, i), h.item(j, i + 1), h.item(j + 1, i), h.item(j + 1, i + 1)
        tx, tz = fx - i, fz - j
        if h00 == h00 and h10 == h10 and h01 == h01 and h11 == h11:      # alle vier Zellen Fahrbahn
            height = (h00 * (1 - tx) + h10 * tx) * (1 - tz) + (h01 * (1 - tx) + h11 * tx) * tz
            sx = ((h10 - h00) * (1 - tz) + (h11 - h01) * tz) / RES
            sz = ((h01 - h00) * (1 - tx) + (h11 - h10) * tx) / RES
            return height, sx, sz
        v = h.item(j + (tz >= 0.5), i + (tx >= 0.5))                       # am Rand: nächste Zelle
        return (v, None, None) if v == v else (None, None, None)

    def _validate(self):
        dists = [g.dist for g in self.checkpoints]
        if dists != sorted(dists) or (dists and (dists[0] <= 0 or dists[-1] >= self.finish_gate.dist)):
            raise ValueError(f"{self.id}: Checkpoints müssen in Fahrtrichtung zwischen Start und Ziel liegen")
        if self.start_y is None:
            raise ValueError(f"{self.id}: Am Start ist keine Fahrbahn")

    # ---------- Abfragen ----------
    def progress_at(self, x, z, hint=None, window=40):
        """(Meter ab Startlinie, Segment-Index, Tangente, seitl. Abstand) für eine Position."""
        n = len(self.seg_dir)
        idx = np.arange(n) if hint is None else np.arange(max(0, hint - window), min(n, hint + window + 1))
        a, e, L = self.center[idx], self.seg_dir[idx], self.seg_len[idx]
        t = np.clip((x - a[:, 0]) * e[:, 0] + (z - a[:, 1]) * e[:, 1], 0.0, L)
        px, pz = a[:, 0] + e[:, 0] * t, a[:, 1] + e[:, 1] * t
        k = int(np.argmin((px - x) ** 2 + (pz - z) ** 2))
        i = int(idx[k])
        dist = self.cum[i] + t[k] - self._dist_offset
        ex, ez = e[k]
        lateral = (x - px[k]) * ez - (z - pz[k]) * ex
        return float(dist), i, (float(ex), float(ez)), float(lateral)

    def segment_at(self, dist):
        d = min(max(dist + self._dist_offset, 0.0), self.cum[-1])
        i = min(int(np.searchsorted(self.cum, d, side="right")) - 1, len(self.seg_dir) - 1)
        return i, d - self.cum[i]

    def point_at(self, dist):
        """Punkt + Tangente auf der Mittellinie, `dist` Meter nach der Startlinie."""
        i, rest = self.segment_at(dist)
        return self.center[i] + self.seg_dir[i] * rest, self.seg_dir[i]

    def surface_height_at(self, dist, lateral):
        """Höhe der Fahrbahnoberfläche `dist` m nach dem Start, `lateral` m neben der Mitte (+ = rechts).
        Berücksichtigt die Querneigung (Steilkurven). Neben der Fahrbahn: Höhe der nächsten Kante."""
        i, rest = self.segment_at(dist)
        t = rest / self.seg_len[i]
        y = self.center_y[i] * (1 - t) + self.center_y[i + 1] * t
        tilt = self.center_tilt[i] * (1 - t) + self.center_tilt[i + 1] * t
        half = (self.center_width[i] * (1 - t) + self.center_width[i + 1] * t) / 2
        return float(y + tilt * max(-half, min(half, lateral)))

    def height_at(self, dist):
        """Höhe der Mittellinie (auch über Lücken, dort zwischen Absprung und Landung)."""
        i, rest = self.segment_at(dist)
        t = rest / self.seg_len[i]
        return float(self.center_y[i] * (1 - t) + self.center_y[i + 1] * t)
