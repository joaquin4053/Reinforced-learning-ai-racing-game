"""Geister / Replays: Fahrten als JSON in ghosts/ speichern und wieder laden.

Eine Geist-Datei:
{
  "meta": {
    "format": 1, "physics_version": 6, "rules_version": 1, "track_version": 2, "sensor_version": 1,
    "map": "map01", "source": "player" | "ai", "attempt": 200 (nur KI), "driver": "Autopilot",
    "result": "finished" | "crashed", "end_reason": "wall" | ..., "finish_time": 41.23,
    "progress_pct": 63.5, "wall_hits": 3, "splits": [9.8, 18.2, ...], "dt": 0.01667, "created": "2026-10-13 18:00:00"
  },
  "frames": [[x, z, heading], ...]   eine Zeile pro Physik-Schritt
}
"""
import json
import math
import time
from pathlib import Path

from .car import DT
from .sim import END_REASONS, PHYSICS_VERSION, RULES_VERSION, SENSOR_VERSION

GHOST_DIR = Path("ghosts")
TRACK_DIR = Path("tracks")
RECORDS_FILE = Path("saves/bestzeiten.json")
FORMAT = 1


def save_ghost(sim, source, attempt=None, driver=None, folder=GHOST_DIR):
    """Fahrt aus einer Sim (mit record=True) als Geist speichern. Gibt den Pfad zurück."""
    if source not in ("player", "ai"):
        raise ValueError("source muss 'player' oder 'ai' sein")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    meta = {
        "format": FORMAT,
        "physics_version": PHYSICS_VERSION,
        "rules_version": RULES_VERSION,
        "track_version": sim.track.version,
        "sensor_version": SENSOR_VERSION,
        "map": sim.track.id,
        "source": source,
        "attempt": attempt,
        "driver": driver,
        "result": sim.result,
        "end_reason": sim.end_reason,
        "finish_time": sim.finish_time,
        "progress_pct": round(sim.progress_pct, 2),
        "wall_hits": sim.wall_hits,
        "splits": [round(s, 4) for s in sim.splits],
        "dt": DT,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    frames = [[round(x, 3), round(z, 3), round(h, 4)] for x, z, h in sim.history]
    if source == "ai" and attempt is not None:
        name = f"{sim.track.id}_ai_{attempt:06d}.json"
    else:
        name = f"{sim.track.id}_{source}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path = folder / name
    path.write_text(json.dumps({"meta": meta, "frames": frames}, separators=(",", ":")))
    return path


class Ghost:
    def __init__(self, path):
        self.path = Path(path)
        data = json.loads(self.path.read_text())
        self.meta = data["meta"]
        self.frames = data["frames"]

    @property
    def label(self):
        m = self.meta
        if m["source"] == "ai":
            name = f"KI · Versuch {m['attempt']}" if m.get("attempt") is not None else "KI"
            return f"{name} ({m['driver']})" if m.get("driver") else name
        return m.get("driver") or "Spieler"

    @property
    def compatible(self):
        """Passt der Geist zur aktuellen Physik und zur aktuellen Version der Map?"""
        return _current_versions(self.meta, self.meta["map"])

    @property
    def result_text(self):
        m = self.meta
        if m["result"] == "finished":
            hits = m.get("wall_hits", 0)
            return f"{m['finish_time']:.2f} s" + (f" ({hits}× Wand)" if hits else "")
        if m["result"] == "truncated":
            return f"Abgebrochen bei {m['progress_pct']:.0f} % (Zeitlimit)"
        return f"Crash bei {m['progress_pct']:.0f} % ({END_REASONS.get(m['end_reason'], m['end_reason'])})"

    @property
    def last_frame(self):
        return len(self.frames) - 1

    def state_at(self, frame):
        """(x, z, heading) im Physik-Schritt `frame`. Nach dem Ende bleibt der Geist stehen."""
        return self.frames[max(0, min(frame, self.last_frame))]

    def speed_at(self, frame):
        """Tempo in m/s, aus zwei aufeinanderfolgenden Positionen berechnet."""
        f = max(1, min(frame, self.last_frame))
        (x0, z0, _), (x1, z1, _) = self.frames[f - 1], self.frames[f]
        return math.hypot(x1 - x0, z1 - z0) / self.meta["dt"]

    def split(self, index):
        """Zeit am Checkpoint `index` (0 = erster), oder None wenn nie erreicht."""
        s = self.meta["splits"]
        return s[index] if index < len(s) else None


def list_ghosts(map_id=None, folder=GHOST_DIR):
    """Alle Geister (optional nur einer Map), z.B. für das spätere Auswahl-Menü."""
    ghosts = [Ghost(p) for p in sorted(Path(folder).glob("*.json"))]
    return [g for g in ghosts if map_id is None or g.meta["map"] == map_id]


def track_version(map_id):
    """Aktuelle Versionsnummer einer Map (aus tracks/<map>.json), None wenn es die Map nicht gibt."""
    try:
        return int(json.loads((TRACK_DIR / f"{map_id}.json").read_text()).get("version", 1))
    except FileNotFoundError:
        return None


def _current_versions(info, map_id):
    """Wurde etwas (Geist oder Bestzeit) mit der aktuellen Physik, den aktuellen Regeln
    UND der aktuellen Map-Version gefahren? (Regel R8: sonst nicht vergleichen)"""
    return (info.get("physics_version") == PHYSICS_VERSION
            and info.get("rules_version", 1) == RULES_VERSION
            and info.get("track_version", 1) == track_version(map_id))


# ---------- Bestzeiten des Spielers (getrennt von den Geistern) ----------
def load_records():
    try:
        return json.loads(RECORDS_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def best_time(map_id):
    """Aktuelle Bestzeit einer Map, oder None.

    Zählt nur, wenn sie mit der aktuellen Physik und Map-Version gefahren wurde
    und der Geist noch existiert.
    """
    record = load_records().get(map_id)
    if record is None or not _current_versions(record, map_id):
        return None
    if not Path(record["ghost"]).exists():
        return None
    return record


def submit_time(map_id, finish_time, ghost_path):
    """Neue Bestzeit speichern, falls besser. Gibt True zurück bei neuem Rekord.

    Rekorde aus älteren Physik- oder Map-Versionen werden aufgehoben (unter "alte_rekorde"),
    aber nicht verglichen.
    """
    records = load_records()
    old = records.get(map_id)
    current = old is not None and _current_versions(old, map_id)
    if current and old["time"] <= finish_time:
        return False
    if old is not None and not current:
        records.setdefault("alte_rekorde", []).append({"map": map_id, **old})
    records[map_id] = {"time": finish_time, "ghost": str(ghost_path), "physics_version": PHYSICS_VERSION,
                       "rules_version": RULES_VERSION, "track_version": track_version(map_id),
                       "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    RECORDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RECORDS_FILE.write_text(json.dumps(records, indent=2))
    return True
