"""Alle Geister in ghosts/ auflisten (Vorstufe für das spätere Auswahl-Menü).

    python tools/list_ghosts.py [map01]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from game.ghost import list_ghosts  # noqa: E402

for g in list_ghosts(sys.argv[1] if len(sys.argv) > 1 else None):
    warn = "" if g.compatible else "  [alte Version]"
    print(f"{g.path}   {g.meta['map']:<8} {g.label:<32} {g.result_text}{warn}")
