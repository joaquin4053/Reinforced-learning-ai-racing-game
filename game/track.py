"""Strecke aus einer JSON-Datei in tracks/ laden.

Format (siehe tracks/map01.json):
{
  "name": "Erste Runde",
  "version": 2,                bei jeder Änderung der Map erhöhen -> alte Geister/Bestzeiten zählen nicht mehr
  "width": 20,                 Streckenbreite in Metern
  "loop": true,                Rundkurs? (false = Start und Ziel an verschiedenen Stellen)
  "laps": 1,                   Runden (nur bei loop)
  "smooth": 8,                 Zwischenpunkte pro Abschnitt für weiche Kurven (1 = eckig)
  "points": [[x, z], ...],     Mittellinie in Fahrtrichtung (Meter)
  "start": 0,                  Index in points: Startlinie
  "checkpoints": [4, 9, 14],   Indizes in points, in Fahrtreihenfolge
  "finish": 0                  Index in points: Ziellinie (bei Rundkurs = start)
}

Koordinaten: flache Ebene (x, z), +z = Norden. Heading h: Vorwärts = (sin h, cos h).
"""
import json
import math
from pathlib import Path

import numpy as np

START_OFFSET = 6.0  # das Auto steht so viele Meter nach der Startlinie


class Gate:
    """Eine Linie quer über die Strecke (Start, Checkpoint oder Ziel)."""

    def __init__(self, center, tangent, half_width, dist, name):
        self.center = center          # (x, z)
        self.tangent = tangent        # Fahrtrichtung (Einheitsvektor)
        left = np.array([-tangent[1], tangent[0]])
        self.a = center + left * half_width
        self.b = center - left * half_width
        self.dist = dist              # Meter ab Startlinie entlang der Mittellinie
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


def _catmull_rom(points, sub, loop):
    """Kontrollpunkte -> dichte, weiche Linie. Gibt (Punkte, Index-Zuordnung) zurück."""
    n = len(points)
    out, index_of = [], {}
    segments = n if loop else n - 1
    for i in range(segments):
        p0 = points[(i - 1) % n] if loop or i > 0 else points[0]
        p1, p2 = points[i], points[(i + 1) % n]
        p3 = points[(i + 2) % n] if loop or i + 2 < n else points[-1]
        index_of[i] = len(out)
        for k in range(sub):
            t = k / sub
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    if loop:
        out.append(out[0])           # geschlossen: letzter Punkt = erster Punkt
        index_of[n] = len(out) - 1
    else:
        index_of[n - 1] = len(out)
        out.append(points[-1])
    return np.array(out), index_of


class Track:
    def __init__(self, data, path=None):
        self.path = str(path) if path else None
        self.id = Path(path).stem if path else data.get("name", "map")
        self.name = data.get("name", self.id)
        self.version = int(data.get("version", 1))
        self.width = float(data.get("width", 14))
        self.is_loop = bool(data.get("loop", True))
        self.laps = int(data.get("laps", 1)) if self.is_loop else 1
        pts = np.array(data["points"], dtype=np.float64)
        sub = max(1, int(data.get("smooth", 8)))

        # Mittellinie: dichte Punkte + gefahrene Meter bis zu jedem Punkt
        self.center, index_of = _catmull_rom(pts, sub, self.is_loop)
        seg = np.diff(self.center, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.seg_dir = seg / self.seg_len[:, None]

        # Startlinie bei "start": Strecke so verschieben, dass dort Meter 0 ist
        start_i = index_of[int(data.get("start", 0))]
        self.start_index = start_i
        self._dist_offset = self.cum[start_i]
        self.length = self.cum[-1] if self.is_loop else self.cum[-1] - self._dist_offset

        half = self.width / 2
        self.start_gate = self._gate(start_i, "Start")
        self.checkpoints = [self._gate(index_of[i], f"CP {k + 1}")
                            for k, i in enumerate(data.get("checkpoints", []))]
        finish_i = index_of[int(data.get("finish", 0 if self.is_loop else len(pts) - 1))]
        self.finish_gate = self._gate(finish_i, "Ziel")
        if self.is_loop:
            self.finish_gate.dist = self.length

        t = self.start_gate.tangent
        self.start_heading = math.atan2(t[0], t[1])
        sx, sz = self.start_gate.center + t * START_OFFSET
        self.start_pos = (float(sx), float(sz))

        self._build_walls(half)
        self._validate(data)

    @staticmethod
    def load(path):
        path = Path(path)
        return Track(json.loads(path.read_text()), path)

    # ---------- Aufbau ----------
    def _tangent(self, i):
        n = len(self.seg_dir)
        if self.is_loop:
            d = self.seg_dir[(i - 1) % n] + self.seg_dir[i % n]
        else:
            d = self.seg_dir[max(i - 1, 0)] + self.seg_dir[min(i, n - 1)]
        return d / np.hypot(*d)

    def _gate(self, i, name):
        dist = (self.cum[i] - self._dist_offset) % self.cum[-1] if self.is_loop else self.cum[i] - self._dist_offset
        return Gate(self.center[i].copy(), self._tangent(i), self.width / 2 + 1.0, dist, name)

    def _build_walls(self, half):
        tangents = np.array([self._tangent(i) for i in range(len(self.center))])
        left = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
        self.left_edge = self.center + left * half
        self.right_edge = self.center - left * half
        walls = []
        for edge in (self.left_edge, self.right_edge):
            walls += [(*a, *b) for a, b in zip(edge[:-1], edge[1:])]
        if not self.is_loop:  # Strecke vorne und hinten zumachen
            walls.append((*self.left_edge[0], *self.right_edge[0]))
            walls.append((*self.left_edge[-1], *self.right_edge[-1]))
        self.walls = np.array(walls, dtype=np.float64)  # (S, 4): x1 z1 x2 z2

    def _validate(self, data):
        """Hilfreiche Fehlermeldungen für selbstgebaute Maps."""
        dists = [g.dist for g in self.checkpoints]
        if dists != sorted(dists) or (dists and (dists[0] <= 0 or dists[-1] >= self.finish_gate.dist)):
            raise ValueError(f"{self.id}: Checkpoints müssen in Fahrtrichtung zwischen Start und Ziel liegen")
        # Überlappen sich zwei Streckenteile? (Abstand Mittellinie < Breite + Wand)
        c = self.center[:-1] if self.is_loop else self.center
        d = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])
        along = np.abs(self.cum[:len(c), None] - self.cum[None, :len(c)])
        if self.is_loop:
            along = np.minimum(along, self.cum[-1] - along)
        close = (d < self.width + 2.0) & (along > self.width * 2.5)
        if close.any():
            i = int(np.argwhere(close)[0][0])
            x, z = self.center[i]
            raise ValueError(f"{self.id}: Strecke überlappt sich bei ca. ({x:.0f}, {z:.0f}) – Punkte weiter auseinander legen")
        # Zu enge Kurven -> innere Wand würde sich falten
        a, b = self.seg_dir[:-1], self.seg_dir[1:]
        turn = np.abs(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], np.sum(a * b, axis=1)))
        radius = self.seg_len[1:] / np.maximum(turn, 1e-9)
        if radius.min() < self.width / 2 + 1.0:
            x, z = self.center[int(radius.argmin()) + 1]
            raise ValueError(f"{self.id}: Kurve bei ca. ({x:.0f}, {z:.0f}) ist zu eng für Breite {self.width:.0f} m")

    # ---------- Abfragen ----------
    def progress_at(self, x, z, hint=None, window=40):
        """Fortschritt für eine Position: (Meter ab Startlinie, Segment-Index, Tangente, seitl. Abstand).

        hint = Segment-Index vom letzten Schritt -> nur in der Nähe suchen. Wichtig, wenn
        die Strecke an sich selbst vorbeiführt: dann wird nicht auf den falschen Teil gesprungen.
        """
        n = len(self.seg_dir)
        if hint is None:
            idx = np.arange(n)
        elif self.is_loop:
            idx = np.arange(hint - window, hint + window + 1) % n
        else:
            idx = np.arange(max(0, hint - window), min(n, hint + window + 1))
        a = self.center[idx]
        e = self.seg_dir[idx]
        L = self.seg_len[idx]
        t = np.clip((x - a[:, 0]) * e[:, 0] + (z - a[:, 1]) * e[:, 1], 0.0, L)
        px, pz = a[:, 0] + e[:, 0] * t, a[:, 1] + e[:, 1] * t
        k = int(np.argmin((px - x) ** 2 + (pz - z) ** 2))
        i = int(idx[k])
        dist = self.cum[i] + t[k] - self._dist_offset
        if self.is_loop:
            dist %= self.cum[-1]
        ex, ez = e[k]
        lateral = (x - px[k]) * ez - (z - pz[k]) * ex  # >0 = rechts der Mitte
        return float(dist), i, (float(ex), float(ez)), float(lateral)

    def segment_at(self, dist):
        """Index des Mittellinien-Abschnitts `dist` Meter nach der Startlinie (+ Meter seit Abschnittsbeginn)."""
        d = dist + self._dist_offset
        d = d % self.cum[-1] if self.is_loop else min(max(d, 0.0), self.cum[-1])
        i = min(int(np.searchsorted(self.cum, d, side="right")) - 1, len(self.seg_dir) - 1)
        return i, d - self.cum[i]

    def point_at(self, dist):
        """Punkt + Tangente auf der Mittellinie, `dist` Meter nach der Startlinie."""
        i, rest = self.segment_at(dist)
        return self.center[i] + self.seg_dir[i] * rest, self.seg_dir[i]
