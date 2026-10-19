"""Alte Trainingsläufe löschen oder nur ihre Versuchs-Dateien entfernen. (Plan-Schritt 5)

    .venv/bin/python tools/aufraeumen.py                        # Übersicht, nichts wird gelöscht
    .venv/bin/python tools/aufraeumen.py v2_test1 --episoden    # nur episodes/ löschen
    .venv/bin/python tools/aufraeumen.py v2_test1 --alles       # ganzen Lauf löschen
    .venv/bin/python tools/aufraeumen.py lauf1 lauf2 --alles    # mehrere auf einmal

--episoden: Snapshots (models/), Bestzeit-Modell, summary.csv und alles andere bleiben.
Vor dem Löschen wird immer angezeigt, was wegkommt, und erst nach Eingabe von "ja" gelöscht.
Läufe, deren Training gerade läuft, werden nicht angefasst.
"""
import argparse
import shutil
import sys
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def size(path):
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.is_dir() else 0


def running_trainings():
    """Namen der Läufe, für die gerade ein train.py läuft."""
    names = set()
    for p in psutil.process_iter(["cmdline"]):
        cmd = p.info.get("cmdline") or []
        if any(c.endswith("train.py") for c in cmd) and "--name" in cmd:
            i = cmd.index("--name")
            if i + 1 < len(cmd):
                names.add(cmd[i + 1])
    return names


def overview():
    runs = sorted(p for p in RUNS.iterdir() if p.is_dir()) if RUNS.exists() else []
    if not runs:
        print("Keine Läufe in runs/.")
        return
    busy = running_trainings()
    print(f"{'Lauf':<20} {'gesamt':>10} {'davon episodes/':>16}")
    for r in runs:
        ep = r / "episodes"
        n = len(list(ep.glob("*.npz")))
        flag = "   (Training läuft)" if r.name in busy else ""
        print(f"{r.name:<20} {size(r) / 1e6:7.1f} MB {size(ep) / 1e6:9.1f} MB ({n} Versuche){flag}")
    print("\nLöschen: aufraeumen.py <lauf> --episoden   oder   aufraeumen.py <lauf> --alles")


def main():
    ap = argparse.ArgumentParser(description="Trainingsläufe aufräumen")
    ap.add_argument("namen", nargs="*", help="Läufe in runs/")
    what = ap.add_mutually_exclusive_group()
    what.add_argument("--episoden", action="store_true", help="nur episodes/ löschen")
    what.add_argument("--alles", action="store_true", help="ganzen Lauf löschen")
    args = ap.parse_args()

    if not args.namen:
        overview()
        return
    if not (args.episoden or args.alles):
        sys.exit("Bitte angeben, was weg soll: --episoden (nur Versuchs-Dateien) oder --alles (ganzer Lauf).")
    busy = running_trainings()
    targets = []
    for name in args.namen:
        run = RUNS / name
        if not run.is_dir() or run.parent != RUNS:
            sys.exit(f"Lauf '{name}' gibt es nicht in runs/.")
        if name in busy:
            sys.exit(f"Das Training '{name}' läuft gerade. Erst pausieren (Ctrl+C), dann aufräumen.")
        path = run / "episodes" if args.episoden else run
        if path.exists():
            targets.append(path)

    if not targets:
        print("Nichts zu löschen.")
        return
    print("Wird gelöscht:")
    total = 0
    for p in targets:
        b = size(p)
        total += b
        detail = f"{len(list(p.glob('*.npz')))} Versuchs-Dateien" if args.episoden else "ganzer Lauf"
        print(f"  {p.relative_to(ROOT)}/   {b / 1e6:.1f} MB   ({detail})")
    if args.episoden:
        print("Bleiben: models/ (Snapshots), best_model.zip, summary.csv, evals.csv, checkpoint/ und der Rest.")
    print(f"Zusammen {total / 1e6:.1f} MB. Das kann nicht rückgängig gemacht werden.")
    try:
        answer = input('Wirklich löschen? "ja" eingeben: ').strip().lower()
    except EOFError:
        answer = ""
    if answer != "ja":
        print("Abgebrochen, nichts gelöscht.")
        return
    for p in targets:
        shutil.rmtree(p)
        print(f"Gelöscht: {p.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
