"""Wie viel Speicher belegen die Trainingsläufe, und wie viel kommt pro 1 Million Schritte dazu? (Plan-Schritt 5)

    .venv/bin/python tools/speicher.py              # alle Läufe in runs/
    .venv/bin/python tools/speicher.py v2_test1     # nur diese(n)

Ziel: ein ganzer Lauf mit 5 Millionen Schritten bleibt unter 1 GB.
Hochrechnung = feste Teile (Fortsetzungspunkt, Bestzeit-Modell, ...) + 5 × (Zuwachs pro 1 Mio. Schritte).
Der Zuwachs von Versuchen und Snapshots wird nur über den Teil des Laufs gemessen, der schon aufgezeichnet wurde.
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
ZIEL_SCHRITTE = 5_000_000
ZIEL_BYTES = 1e9

# (Name, Dateien/Ordner im Lauf, wächst mit dem Training?)
TEILE = [
    ("Versuche (episodes/)", ["episodes"], "aufzeichnung"),
    ("Snapshots (models/)", ["models"], "aufzeichnung"),
    ("Fortsetzungspunkt", ["checkpoint", "checkpoint.alt", "checkpoint.neu"], "fest"),
    ("Bestzeit-Modell + Geister", ["best_model.zip", "ghosts"], "lauf"),
    ("Zeitleiste (versionen/)", ["versionen"], "lauf"),
    ("Alte Snapshots (snapshots/)", ["snapshots"], "lauf"),
    ("TensorBoard (tb/)", ["tb"], "lauf"),
    ("Tabellen, Protokolle", ["summary.csv", "evals.csv", "ereignisse.csv", "summary_alt.csv", "training.log",
                              "watch.log"], "lauf"),
    ("Konfiguration, Code, Live", ["config.json", "code.tar.gz", "live.json", "live_model.zip", "rekorde.json"],
     "fest"),
]


def size(path):
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return 0


def mb(b):
    return f"{b / 1e6:8.1f} MB"


def _n(x):
    return f"{x:,}".replace(",", "'")


def _rows(path):
    try:
        with open(path) as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def steps_info(run):
    """(Schritte insgesamt, Schritte seit Beginn der Aufzeichnung)."""
    total = 0
    for name in ("summary.csv", "evals.csv", "ereignisse.csv", "summary_alt.csv"):
        rows = _rows(run / name)
        if rows and rows[-1].get("schritte", "").isdigit():
            total = max(total, int(rows[-1]["schritte"]))
    # Trainingsschritte mit Aufzeichnung = Summe der Fahrten (Auswertungen zählen nicht als Trainingsschritte,
    # ihre Dateien schon: so ist der Zuwachs pro 1 Mio. Trainingsschritte ehrlich gerechnet)
    rec = _rows(run / "summary.csv")
    recorded = 0
    if rec and "versuch" in rec[0]:
        recorded = sum(int(r["fahrt_schritte"] or 0) for r in rec if r["art"] == "training")
    return total, recorded


def other_files(run):
    known = {n for _, names, _ in TEILE for n in names}
    return [p for p in run.iterdir() if p.name not in known]


def report(run):
    total_steps, rec_steps = steps_info(run)
    print(f"\n{run.name}   ({_n(total_steps)} Schritte, davon {_n(rec_steps)} mit Aufzeichnung)")
    print(f"  {'Teil':<30} {'belegt':>11} {'pro 1 Mio. Schritte':>21}")
    fixed, per_m, all_bytes = 0.0, 0.0, 0
    for label, names, kind in TEILE:
        b = sum(size(run / n) for n in names)
        all_bytes += b
        steps = rec_steps if kind == "aufzeichnung" else total_steps
        if kind == "fest":
            fixed += b
            rate = "fest"
        elif steps > 0:
            r = b / steps * 1e6
            per_m += r
            rate = mb(r)
        else:
            rate = "–"
        if b or kind == "aufzeichnung":
            print(f"  {label:<30} {mb(b):>11} {rate:>21}")
    rest = sum(size(p) for p in other_files(run))
    if rest:
        all_bytes += rest
        fixed += rest
        print(f"  {'Sonstiges':<30} {mb(rest):>11} {'fest':>21}")
    print(f"  {'Zusammen':<30} {mb(all_bytes):>11} {mb(per_m):>21}")
    if rec_steps > 0:
        proj = fixed + per_m * ZIEL_SCHRITTE / 1e6
        ok = "unter" if proj < ZIEL_BYTES else "ÜBER"
        print(f"  Hochrechnung {_n(ZIEL_SCHRITTE)} Schritte: {proj / 1e6:.0f} MB ({ok} 1 GB)")
        n_ep = len(list((run / "episodes").glob("*.npz")))
        if n_ep:
            print(f"  {_n(n_ep)} Versuche aufgezeichnet, im Schnitt {size(run / 'episodes') / n_ep / 1e3:.0f} KB "
                  f"pro Versuch, {size(run / 'episodes') / rec_steps:.0f} Byte pro Schritt")
    else:
        print("  Noch keine Aufzeichnung (Lauf von vor Schritt 5) – keine Hochrechnung möglich.")
    return all_bytes


def main():
    ap = argparse.ArgumentParser(description="Speicherbedarf der Trainingsläufe")
    ap.add_argument("namen", nargs="*", help="Läufe (Standard: alle in runs/)")
    args = ap.parse_args()
    runs = [RUNS / n for n in args.namen] if args.namen else sorted(p for p in RUNS.iterdir() if p.is_dir())
    missing = [r.name for r in runs if not r.is_dir()]
    if missing:
        sys.exit(f"Nicht gefunden in runs/: {', '.join(missing)}")
    total = sum(report(r) for r in runs)
    if len(runs) > 1:
        print(f"\nAlle Läufe zusammen: {total / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
