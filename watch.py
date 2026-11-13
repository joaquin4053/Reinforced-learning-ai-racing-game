"""Der KI beim Fahren zuschauen: Eine gespeicherte Version des Netzes fährt jetzt, deterministisch (ohne Zufall, R6).

    .venv/bin/python watch.py v2_lauf2 --live              # LIVE: immer die neueste Version, folgt dem Training
    .venv/bin/python watch.py v2_lauf2                     # Version der offiziellen Bestzeit
    .venv/bin/python watch.py v2_lauf2 --modell neuester   # jüngste Version (ohne zu folgen)
    .venv/bin/python watch.py v2_lauf2 --modell letzter    # letzter Fortsetzungspunkt
    .venv/bin/python watch.py v2_lauf2 --modell runs/v2_lauf2/versionen/model_0000250000.zip
    .venv/bin/python watch.py v2_lauf2 --gegen-mich        # mit deiner Bestzeit als Geist

Gleiche Oberfläche wie viewer.py (render/viewer_app.py): Die obere Zeitleiste zeigt die Versionen
(eine pro Auswertung, alle 25'000 Schritte; Höhe = Fortschritt, Farbe = Ergebnis), die untere die Fahrt.
Was schon gefahren ist, lässt sich zurückspulen; vorspulen rechnet voraus. Tastenkürzel: ? drücken.
Die KI sieht nur ihre Beobachtung (R5) und fährt mit derselben Simulation wie der Spieler (R3).
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ap = argparse.ArgumentParser(description="Der KI beim Fahren zuschauen")
ap.add_argument("name", help="Name des Trainings (Ordner runs/<name>/)")
ap.add_argument("--modell", default="best", help="best, neuester, letzter oder Pfad zu einer .zip-Datei")
ap.add_argument("--live", action="store_true", help="immer die neueste Version zeigen und dem Training folgen")
ap.add_argument("--gegen-mich", action="store_true", help="eigene Bestzeit als Geist mitfahren lassen")
ap.add_argument("--ghost", action="append", default=[], help="weitere Geist-Datei (mehrfach möglich)")
ap.add_argument("--tempo", type=float, default=1, choices=(0.5, 1, 2, 5, 10, 50))
ap.add_argument("--foto", nargs=2, metavar=("DATEI", "SEKUNDEN"), help=argparse.SUPPRESS)   # Test: Bild speichern
ap.add_argument("--tasten", default="", help=argparse.SUPPRESS)   # Test: Tasten/Maus wie bei viewer.py
args = ap.parse_args()

run_dir = Path(args.name) if Path(args.name).is_dir() else ROOT / "runs" / args.name
if not (run_dir / "config.json").exists():
    sys.exit(f"Training '{args.name}' nicht gefunden (runs/{args.name}/config.json fehlt).")
cfg = json.loads((run_dir / "config.json").read_text())
os.chdir(ROOT)                          # Geister und deine Bestzeit werden relativ zum Projekt gefunden

import torch  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

from game.ghost import best_time  # noqa: E402
from game.track import Track  # noqa: E402
from render.viewer_app import Source, ViewerApp, nr, result_text  # noqa: E402
from rl.env import RaceEnv  # noqa: E402
from rl.playback import SPEEDS, EpisodeIndex, Player, VersionIndex, read_json  # noqa: E402

torch.set_num_threads(1)
if RaceEnv(config=cfg["reward_config"]).versions["sensoren"] != cfg["versionen"]["sensoren"]:
    sys.exit("Die Sensoren haben sich seit dem Training geändert – dieses Modell passt nicht mehr (R8).")

track = Track(cfg["strecke"])           # genau die Strecke dieses Trainings (R8)
track.id = cfg["strecke_id"]
mine = best_time(track.id)
ghosts = list(args.ghost) + ([mine["ghost"]] if args.gegen_mich and mine else [])

# Zusätzliche Modelle, die nicht in versionen/ liegen
cp = next((d for d in (run_dir / "checkpoint", run_dir / "checkpoint.alt") if (d / "model.zip").exists()), None)
extra, start_steps = [], None
if args.modell == "letzter" and cp:
    start_steps = read_json(cp / "state.json").get("schritte", 0)
    extra.append((start_steps, cp / "model.zip"))
elif args.modell not in ("best", "neuester", "letzter"):
    path = Path(args.modell)
    if not path.exists():
        sys.exit(f"Kein Modell gefunden ({args.modell}).")
    try:
        start_steps = int(path.stem.split("_")[1])
    except (IndexError, ValueError):
        start_steps = -1                # unbekannte Datei: ganz vorne einsortieren
    extra.append((start_steps, path))


class ModelSource(Source):
    item, items = "Version", "Versionen"
    filters = ("alle", "gueltige")
    slow_load = True
    title = "KI fährt"
    live_text = "neueste Version, folgt dem Training"
    filter_text = {"alle": "Alle Versionen", "gueltige": "Nur Versionen im Ziel"}
    ghosts_enabled = False              # jeder Geist bräuchte ein eigenes Netz, das jetzt fährt

    @staticmethod
    def name(a):
        """Version nach ihrem Versuch benennen (die Auswertung, bei der sie gespeichert wurde)."""
        return f"Versuch {nr(a.auswertung_nr)}" if a.auswertung_nr else f"Schritt {nr(a.versuch)}"

    def info(self, a):
        text = f"{self.name(a)}  ·  fährt jetzt (ohne Zufall)"
        return text + (f"  ·  Auswertung damals: {result_text(a)}" if a.bekannt else "")

    def hover(self, a):
        return f"{self.name(a)} · " + (result_text(a) if a.bekannt else "ohne Auswertung")

    def missing(self, a):
        return f"{self.name(a)}: Modell lässt sich nicht laden"

    def end_text(self, a, ep):
        info = ep.info or {}
        if info.get("gueltige_runde"):
            t = info["rundenzeit"]
            text = f"ZIEL  {format_time(t)}"
            if self.mine:
                text += f"\nDu: {format_time(self.mine['time'])}  ({t - self.mine['time']:+.2f} s)"
            return text
        return f"{(info.get('ende') or '').upper()}\n{info.get('fortschritt_prozent', 0):.0f} % geschafft"

    def shown(self, player, extra=()):
        a, ep = player.current, player.episode
        if a is None:
            return ["–"]
        lines = [f"Version von {self.name(a)}  ·  gespeichert bei Schritt {nr(a.schritte)}",
                 "fährt jetzt (ohne Zufall)" + (f"  ·  Auswertung damals: {result_text(a)}" if a.bekannt else "")]
        if ep is not None:
            lines.append(f"Diese Fahrt: {format_time(ep.duration)}" + ("" if ep.done else " ...")
                         + f"  ·  Wandberührungen: {ep.env.sim.wall_hits}")
        return lines + self.footer(player)

    def curve_index(self, player):
        if not hasattr(self, "_curves"):
            self._curves = EpisodeIndex(self.run_dir)          # Lernkurve/Rundenzeit: alle Versuche des Laufs
        return self._curves

    def focus_attempt(self, player):
        return player.current.auswertung_nr if player.current else None

    def jump(self, player, versuch):
        """Klick in eine Kurve: die Version, die bei diesem Versuch (oder kurz danach) gespeichert wurde."""
        known = [(i, a) for i, a in enumerate(player.list) if a.auswertung_nr]
        if known:
            i, _ = min(known, key=lambda x: abs(x[1].auswertung_nr - versuch))
            player.select(i)

    def nets(self, player):
        """(Actor, Critic) des Netzes, das gerade fährt."""
        from rl.netview import Actor, Critic
        ep, a = player.episode, player.current
        if ep is None or not hasattr(ep, "policy"):
            return None, None
        if getattr(self, "_nets_for", None) is not ep.policy:
            self._nets_for = ep.policy
            pol = ep.policy.policy
            self._nets_now = (Actor.from_policy(pol, f"Netz von {self.name(a)} (fährt gerade)"), Critic.from_policy(pol))
        return self._nets_now

    def snapshot_path(self, player):
        from rl.netview import nearest_snapshot
        f = self.focus_attempt(player)
        return nearest_snapshot(self.run_dir / "models", f) if f else None

    def result(self, player):
        info = (player.episode.info or {}) if player.episode is not None else {}
        return "Ziel" if info.get("gueltige_runde") else "Crash" if info.get("crash") else ""


from game.sim import format_time  # noqa: E402


def load_policy(path):
    return SAC.load(path, device="cpu")


def make_env():
    return RaceEnv(config=cfg["reward_config"], record=False, wiederholen=cfg["sac"].get("aktion_wiederholen", 1))


index = VersionIndex(run_dir, make_env, load_policy, extra=extra)
player = Player(index)
player.speed_i = SPEEDS.index(args.tempo if args.tempo != 1 else 1)
if not index.attempts:
    sys.exit("Noch keine Versionen. Sie entstehen bei jeder Auswertung (alle 25'000 Schritte).")
if not args.live:
    if args.modell == "best":
        best = read_json(run_dir / "rekorde.json").get("bestzeit_schritte")
        if best is None:
            print("Noch keine offizielle Bestzeit – zeige die neueste Version.")
        start_steps = best if best is not None else index.attempts[-1].versuch
    elif args.modell == "neuester":
        start_steps = index.attempts[-1].versuch
    player.select_versuch(start_steps)

app = ViewerApp(ModelSource(run_dir, cfg, track, ghosts=ghosts, mine=mine), player, test_keys=args.tasten,
                photo=args.foto)
update, input, text_input = app.update, app.input, app.text_input
app.run()
