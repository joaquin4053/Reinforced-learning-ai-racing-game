"""Zahlen für das Dashboard (Plan-Schritt 8), ohne Grafik, damit sie testbar sind.

Alles wird aus den Dateien des Laufs gelesen (summary.csv, evals.csv, rekorde.json) bzw. aus dem gerade
gezeigten Versuch – nichts wird neu gerechnet oder trainiert.
"""
import csv

import numpy as np


def moving_average(values, window=50):
    """Gleitender Durchschnitt über die letzten `window` Werte (am Anfang über so viele, wie es gibt)."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return v
    c = np.cumsum(np.r_[0.0, v])
    i = np.arange(1, len(v) + 1)
    lo = np.maximum(0, i - window)
    return (c[i] - c[lo]) / (i - lo)


def reward_parts(rewards, weights, result):
    """Reward pro Schritt in die vier Bestandteile von R7 zerlegen.

    Gespeichert ist nur die Summe je Schritt. Zeitstrafe = -k_zeit in jedem Schritt, Crash-Strafe und
    Ziel-Bonus nur im letzten Schritt (je nach Ergebnis), der Rest ist Fortschritt (neuer Höchststand).
    -> dict mit Arrays gleicher Länge: fortschritt, zeit, crash, ziel, summe
    """
    r = np.asarray(rewards, dtype=np.float64)
    n = len(r)
    zeit = np.full(n, -float(weights["k_zeit"]))
    crash, ziel = np.zeros(n), np.zeros(n)
    if n:
        if result == "Crash":
            crash[-1] = -float(weights["k_crash"])
        elif result == "Ziel":
            if "k_ziel_pro_sekunde" in weights:                 # Reward v2: nach Rundenzeit (R7)
                limit = float(weights.get("zeitlimit", 180.0))
                ziel[-1] = float(weights["k_ziel_pro_sekunde"]) * max(0.0, limit - n / 60.0)
            else:
                ziel[-1] = float(weights["k_ziel"])
    return {"fortschritt": r - zeit - crash - ziel, "zeit": zeit, "crash": crash, "ziel": ziel, "summe": r}


def official_best(attempts):
    """Treppe der offiziellen Bestzeit: [(Versuch, Zeit)] bei jeder Verbesserung, nur aus Auswertungen (R6)."""
    out, best = [], None
    for a in attempts:
        if a.auswertung and a.gueltig and a.rundenzeit is not None and (best is None or a.rundenzeit < best):
            best = a.rundenzeit
            out.append((a.versuch, best))
    return out


def share_valid(attempts, n=100):
    """(Anteil gültiger Runden in %, Anzahl) der letzten n Versuche."""
    last = attempts[-n:]
    if not last:
        return 0.0, 0
    return 100.0 * sum(a.gueltig for a in last) / len(last), len(last)


def training_seconds(run_dir):
    """Wie lange das Training gelaufen ist (Wanduhr, ohne Pausen): grösste laufzeit_s in evals/ereignisse."""
    best = 0.0
    for name in ("evals.csv", "ereignisse.csv"):
        try:
            with open(run_dir / name) as f:
                for row in csv.DictReader(f):
                    try:
                        best = max(best, float(row.get("laufzeit_s") or 0))
                    except ValueError:
                        pass
        except OSError:
            pass
    return best
