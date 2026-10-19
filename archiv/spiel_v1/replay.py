"""Einen Geist als Video rendern (Kinokamera).

    python replay.py ghosts/map01_ai_000200.json [--label "Versuch #200"] [--out videos/x.mp4]
                     [--no-rays] [--window] [--seed 3] [--max-seconds 20]

Ohne Pfad wird der neueste Geist aus ghosts/ genommen.
"""
import argparse
from pathlib import Path

from render.recorder import render_video

ap = argparse.ArgumentParser()
ap.add_argument("ghost", nargs="?")
ap.add_argument("--out")
ap.add_argument("--label", help="Text unten links (Standard: Name des Geists)")
ap.add_argument("--no-rays", action="store_true")
ap.add_argument("--window", action="store_true", help="sichtbares Fenster statt offscreen")
ap.add_argument("--seed", type=int, default=0, help="andere Kamera-Abfolge")
ap.add_argument("--max-seconds", type=float, help="nur die ersten N Sekunden rendern")
args = ap.parse_args()

ghost = args.ghost or str(max(Path("ghosts").glob("*.json"), key=lambda p: p.stat().st_mtime))
out = render_video(ghost, args.out, args.label, show_rays=not args.no_rays,
                   offscreen=not args.window, seed=args.seed, max_seconds=args.max_seconds)
print("Video gespeichert:", out)
