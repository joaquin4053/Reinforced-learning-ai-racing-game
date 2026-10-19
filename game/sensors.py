"""Sensoren der KI – nur Fahrersicht (Regel R5).

Die KI bekommt einen Vektor aus Zahlen (die "Beobachtung"). Reihenfolge und Bedeutung stehen in
LAYOUT und als Tabelle in CLAUDE.md. Einstellungen in config.toml unter [sensoren].
Jede Änderung hier oder in [sensoren] erhöht die Sensor-Version (Regel R8).

Strahlen (Spiel v2): Die Fahrbahn schwebt frei. Jeder Strahl läuft waagrecht vom Auto aus über die
Fahrbahn und meldet, wo die Streckenkante beginnt: Ende der Fahrbahn (Abgrund, Lücke) oder ein
Höhensprung (Kante einer Rampe, Seite eines Blocks). Nichts hinter der Reichweite.
"""
import math

import numpy as np

from .config import CONFIG
from .track import RES

_S = CONFIG["sensoren"]
RAY_COUNT = _S["anzahl_strahlen"]
RAY_RANGE = _S["reichweite"]                                     # m
RAY_STEP = _S["schrittweite"]                                    # m
EDGE = _S["kante"]                                               # m
SPEED_NORM = _S["tempo_norm"]                                    # m/s
VERTICAL_NORM = _S["vertikal_norm"]                              # m/s
_half = _S["winkel_bereich"] / 2
RAY_ANGLES_DEG = np.linspace(-_half, _half, RAY_COUNT)           # links (-) bis rechts (+)
RAY_ANGLES = np.radians(RAY_ANGLES_DEG)
_D = np.arange(1, int(round(RAY_RANGE / RAY_STEP)) + 1) * RAY_STEP   # Abtastpunkte entlang eines Strahls
PITCH_NORM = math.pi / 4                                         # 45° = 1


def _layout():
    """(Name, Bedeutung, kleinster Wert, grösster Wert) für jeden Eintrag der Beobachtung."""
    def side(a):
        return "geradeaus" if abs(a) < 1e-9 else ("links" if a < 0 else "rechts")

    rows = [(f"strahl_{i}", f"Abstand zur Streckenkante, Strahl {abs(a):.1f}° {side(a)} "
             f"(0 = direkt an der Kante, 1 = nichts in {RAY_RANGE:.0f} m)", 0.0, 1.0)
            for i, a in enumerate(RAY_ANGLES_DEG)]
    rows += [
        ("tempo", f"Tempo vorwärts / {SPEED_NORM:.0f} m/s (negativ = rückwärts)", -1.0, 1.0),
        ("winkel_strecke", "Winkel zwischen Fahrtrichtung und Streckenrichtung / 180° (positiv = Auto zeigt nach rechts)",
         -1.0, 1.0),
        ("rutschen", f"Tempo seitwärts / {SPEED_NORM:.0f} m/s (positiv = rutscht nach rechts)", -1.0, 1.0),
        ("lenkung", "aktuelle Lenkradstellung (-1 = voll links, +1 = voll rechts)", -1.0, 1.0),
        ("steigung", "Neigung des Autos nach vorne / 45° (positiv = bergauf)", -1.0, 1.0),
        ("in_der_luft", "1 = Räder haben keinen Bodenkontakt, 0 = am Boden", 0.0, 1.0),
        ("tempo_vertikal", f"Tempo nach oben / {VERTICAL_NORM:.0f} m/s (negativ = fällt)", -1.0, 1.0),
    ]
    return rows


LAYOUT = _layout()
SIZE = len(LAYOUT)
LOW = np.array([row[2] for row in LAYOUT], dtype=np.float32)
HIGH = np.array([row[3] for row in LAYOUT], dtype=np.float32)


def ray_distances(x, z, y, heading, track):
    """Abstände der Sensorstrahlen in Metern (höchstens RAY_RANGE)."""
    ang = heading + RAY_ANGLES
    px = x + np.sin(ang)[:, None] * _D[None, :]
    pz = z + np.cos(ang)[:, None] * _D[None, :]
    ix = np.floor((px - track.x0) / RES).astype(np.int64)
    iz = np.floor((pz - track.z0) / RES).astype(np.int64)
    inside = (ix >= 0) & (iz >= 0) & (ix < track.nx) & (iz < track.nz)
    h = np.full(px.shape, np.nan)
    h[inside] = track.height[iz[inside], ix[inside]]
    prev = np.concatenate([np.full((RAY_COUNT, 1), y), h[:, :-1]], axis=1)
    edge = np.isnan(h) | (np.abs(h - prev) > EDGE)
    first = np.where(edge.any(axis=1), edge.argmax(axis=1), len(_D))
    return np.where(first < len(_D), first * RAY_STEP, RAY_RANGE)


def observe(car, track_dir, rays_m, clip=True):
    """Beobachtung als Vektor (Reihenfolge wie LAYOUT)."""
    vf, vr = car.local_velocity()
    fx, fz = car.forward
    tx, tz = track_dir
    angle = math.atan2(fx * tz - fz * tx, fx * tx + fz * tz)
    values = np.concatenate([rays_m / RAY_RANGE,
                             [vf / SPEED_NORM, angle / math.pi, vr / SPEED_NORM, car.steer,
                              car.pitch / PITCH_NORM, 0.0 if car.grounded else 1.0, car.vy / VERTICAL_NORM]])
    if not clip:
        return values
    return np.clip(values, LOW, HIGH).astype(np.float32)


def markdown_table():
    """Tabelle für CLAUDE.md (Index, Name, Bedeutung, Wertebereich)."""
    lines = ["| Index | Name | Bedeutung | Wertebereich |", "|---|---|---|---|"]
    for i, (name, meaning, low, high) in enumerate(LAYOUT):
        lines.append(f"| {i} | `{name}` | {meaning} | {low:g} bis {high:g} |")
    return "\n".join(lines)
