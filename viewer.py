"""Zuschauer-Programm (Plan-Schritt 6): aufgezeichnete Versuche eines Trainings ansehen wie in einem Videoplayer.

    .venv/bin/python viewer.py                  # neuester Lauf in runs/
    .venv/bin/python viewer.py v2_lauf2         # bestimmter Lauf
    .venv/bin/python viewer.py v2_lauf2 --versuch 500 --filter gueltige

Läuft während oder nach dem Training. Liest nur runs/<name>/summary.csv und episodes/ (Plan-Schritt 5),
rechnet nichts nach und lädt kein Netz -> bremst das Training nicht. Gezeigt wird genau, was gefahren wurde.
Die Oberfläche (render/viewer_app.py) ist dieselbe wie in watch.py. Tastenkürzel: ? drücken.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
VIEWER_NICE = 15                        # Training hat immer Vorrang (Profile: nice 0, 10 oder 15)

ap = argparse.ArgumentParser(description="Aufgezeichnete Versuche ansehen")
ap.add_argument("name", nargs="?", help="Lauf in runs/ oder Ordner (Standard: der zuletzt geänderte Lauf)")
ap.add_argument("--versuch", type=int, help="mit diesem Versuch beginnen (statt live)")
ap.add_argument("--filter", choices=("alle", "auswertungen", "gueltige"), default="alle")
ap.add_argument("--tempo", type=float, default=1, choices=(0.5, 1, 2, 5, 10, 50))
ap.add_argument("--gegen-mich", action="store_true", help="eigene Bestzeit als Geist mitfahren lassen")
ap.add_argument("--ghost", action="append", default=[], help="weitere Geist-Datei (mehrfach möglich)")
ap.add_argument("--foto", nargs=2, metavar=("DATEI", "SEKUNDEN"), help=argparse.SUPPRESS)   # Test: Bild speichern
ap.add_argument("--tasten", default="", help=argparse.SUPPRESS)   # Test: Tasten/Maus, z.B. "p maus:versuche:0.3 klick los"
args = ap.parse_args()

if args.name:
    run_dir = Path(args.name) if Path(args.name).is_dir() else RUNS / args.name   # Name oder Ordner
else:
    runs = [p for p in RUNS.glob("*") if (p / "summary.csv").exists()] if RUNS.exists() else []
    if not runs:
        sys.exit("Keine Läufe mit Aufzeichnung in runs/. Erst trainieren: .venv/bin/python train.py --name ...")
    run_dir = max(runs, key=lambda p: (p / "summary.csv").stat().st_mtime)
if not (run_dir / "config.json").exists():
    sys.exit(f"Lauf '{run_dir.name}' nicht gefunden (runs/{run_dir.name}/config.json fehlt).")
cfg = json.loads((run_dir / "config.json").read_text())
os.chdir(ROOT)                          # Geister und deine Bestzeit werden relativ zum Projekt gefunden

from game.ghost import best_time  # noqa: E402
from game.track import Track  # noqa: E402
from render.viewer_app import Source, ViewerApp  # noqa: E402
from rl.playback import SPEEDS, EpisodeIndex, Player  # noqa: E402

track = Track(cfg["strecke"])           # genau die Strecke dieses Trainings (R8)
track.id = cfg["strecke_id"]
mine = best_time(track.id)
ghosts = list(args.ghost) + ([mine["ghost"]] if args.gegen_mich and mine else [])
index = EpisodeIndex(run_dir)
player = Player(index, mode=args.filter)
player.speed_i = SPEEDS.index(args.tempo if args.tempo != 1 else 1)
if not index.attempts:
    print("Noch keine aufgezeichneten Versuche – der Viewer wartet, bis das Training welche schreibt.")
if args.versuch:
    player.select_versuch(args.versuch)
try:
    os.nice(max(0, VIEWER_NICE - os.nice(0)))
except OSError:
    pass

app = ViewerApp(Source(run_dir, cfg, track, ghosts=ghosts, mine=mine), player, test_keys=args.tasten,
                photo=args.foto)
update, input, text_input = app.update, app.input, app.text_input
app.run()
