"""Sensoren der KI – nur Fahrersicht (Regel R5).

Die KI bekommt einen Vektor aus Zahlen (die "Beobachtung"). Reihenfolge und Bedeutung stehen in
LAYOUT und als Tabelle in CLAUDE.md. Einstellungen in config.toml unter [sensoren].
Jede Änderung hier oder in [sensoren] erhöht die Sensor-Version (Regel R8).
"""
import math

import numpy as np

from .config import CONFIG
from .raycast import cast_rays

_S = CONFIG["sensoren"]
RAY_COUNT = _S["anzahl_strahlen"]
RAY_RANGE = _S["reichweite"]                                     # m
SPEED_NORM = _S["tempo_norm"]                                    # m/s
_half = _S["winkel_bereich"] / 2
RAY_ANGLES_DEG = np.linspace(-_half, _half, RAY_COUNT)           # links (-) bis rechts (+)
RAY_ANGLES = np.radians(RAY_ANGLES_DEG)


def _layout():
    """(Name, Bedeutung, kleinster Wert, größter Wert) für jeden Eintrag der Beobachtung."""
    def side(a):
        return "geradeaus" if abs(a) < 1e-9 else ("links" if a < 0 else "rechts")

    rows = [(f"strahl_{i}", f"Abstand zur Wand, Strahl {abs(a):.1f}° {side(a)} "
             f"(0 = direkt an der Wand, 1 = nichts in {RAY_RANGE:.0f} m)", 0.0, 1.0)
            for i, a in enumerate(RAY_ANGLES_DEG)]
    rows += [
        ("tempo", f"Tempo vorwärts / {SPEED_NORM:.0f} m/s (negativ = rückwärts)", -1.0, 1.0),
        ("winkel_strecke", "Winkel zwischen Fahrtrichtung und Streckenrichtung / 180° (positiv = Auto zeigt nach rechts)",
         -1.0, 1.0),
        ("rutschen", f"Tempo seitwärts / {SPEED_NORM:.0f} m/s (positiv = rutscht nach rechts)", -1.0, 1.0),
        ("lenkung", "aktuelle Lenkradstellung (-1 = voll links, +1 = voll rechts)", -1.0, 1.0),
    ]
    return rows


LAYOUT = _layout()
SIZE = len(LAYOUT)
LOW = np.array([row[2] for row in LAYOUT], dtype=np.float32)
HIGH = np.array([row[3] for row in LAYOUT], dtype=np.float32)


def ray_distances(x, z, heading, walls):
    """Abstände der Sensorstrahlen in Metern (höchstens RAY_RANGE)."""
    return cast_rays(x, z, heading + RAY_ANGLES, walls, RAY_RANGE)


def observe(car, track_dir, rays_m, clip=True):
    """Beobachtung als Vektor (Reihenfolge wie LAYOUT). Alle Werte im Bereich -1..1 bzw. 0..1.

    clip=False liefert die Rohwerte (nur für Prüfungen: ist die Normierung groß genug?).
    """
    vf, vr = car.local_velocity()
    fx, fz = car.forward
    tx, tz = track_dir
    angle = math.atan2(fx * tz - fz * tx, fx * tx + fz * tz)   # >0: Auto zeigt rechts der Strecke
    values = np.concatenate([rays_m / RAY_RANGE,
                             [vf / SPEED_NORM, angle / math.pi, vr / SPEED_NORM, car.steer]])
    if not clip:
        return values
    return np.clip(values, LOW, HIGH).astype(np.float32)


def markdown_table():
    """Tabelle für CLAUDE.md (Index, Name, Bedeutung, Wertebereich)."""
    lines = ["| Index | Name | Bedeutung | Wertebereich |", "|---|---|---|---|"]
    for i, (name, meaning, low, high) in enumerate(LAYOUT):
        lines.append(f"| {i} | `{name}` | {meaning} | {low:g} bis {high:g} |")
    return "\n".join(lines)
