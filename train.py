"""KI trainieren (Plan-Schritt 4): SAC aus Stable-Baselines3, kleines Netz (2 × 64 Neuronen).

    .venv/bin/python train.py --name test1                      # neues Training (läuft, bis Ctrl+C)
    .venv/bin/python train.py --name test1 --resume             # genau dort weitermachen
    .venv/bin/python train.py --name test1 --resume --profil sparsam
    .venv/bin/python train.py --name test1 --minuten 30         # nach 30 min selbst pausieren

Ctrl+C pausiert sauber: Fortsetzungspunkt (Modell, Replay-Buffer, Zähler) wird gespeichert.
Alles landet in runs/<name>/ (siehe rl/run.py). Einstellungen: train_config.toml.
"""
import argparse
import json
import os
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)            # Geister und Bestzeiten werden relativ zum Projekt gefunden

from game.config import CONFIG as GAME_CONFIG  # noqa: E402
from rl.env import RaceEnv, load_reward_config  # noqa: E402
from rl.resources import PROFILES, keep_awake  # noqa: E402
from rl.run import Run, git_info, package_versions  # noqa: E402
from rl.trainer import Trainer  # noqa: E402

TRAIN_CONFIG = ROOT / "train_config.toml"


def current_versions(reward_cfg):
    env = RaceEnv(config=reward_cfg)
    return dict(env.versions), env.track


def new_config(args, train_cfg, reward_cfg):
    versions, track = current_versions(reward_cfg)
    track_file = ROOT / reward_cfg["umgebung"]["strecke"]
    return {
        "name": args.name,
        "erstellt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "versionen": versions,
        "strecke_id": track.id,
        "git": git_info(),
        "pakete": package_versions(),
        "befehl": sys.argv,
        "sac": train_cfg["sac"],
        "ablauf": train_cfg["ablauf"],
        "erfolg": train_cfg["erfolg"],
        "reward_config": reward_cfg,
        "spiel_config": GAME_CONFIG,
        "strecke": json.loads(track_file.read_text()),
    }


def main():
    ap = argparse.ArgumentParser(description="KI trainieren (SAC)")
    ap.add_argument("--name", required=True, help="Name des Trainings, Ordner runs/<name>/")
    ap.add_argument("--resume", action="store_true", help="vorhandenes Training fortsetzen")
    ap.add_argument("--profil", choices=PROFILES, default="normal", help="Leistungsprofil (Standard: normal)")
    ap.add_argument("--max-sps", type=float, help="höchstens so viele Schritte pro Sekunde")
    ap.add_argument("--seed", type=int, default=0, help="Zufalls-Seed (nur bei neuem Training)")
    ap.add_argument("--schritte", type=int, help="pausieren, sobald so viele Schritte erreicht sind")
    ap.add_argument("--minuten", type=float, help="pausieren nach so vielen Minuten (diese Sitzung)")
    args = ap.parse_args()

    with open(TRAIN_CONFIG, "rb") as f:
        train_cfg = tomllib.load(f)
    run = Run(args.name)

    if args.resume:
        if not run.exists:
            sys.exit(f"Es gibt kein Training '{args.name}' in runs/. Ohne --resume neu starten.")
        cfg = run.load_config()
        reward_cfg = cfg["reward_config"]
        versions, _ = current_versions(reward_cfg)
        changed = {k: (cfg["versionen"].get(k), v) for k, v in versions.items() if cfg["versionen"].get(k) != v}
        if changed:    # R8: Ergebnisse verschiedener Versionen nie mischen
            text = ", ".join(f"{k} v{a} -> v{b}" for k, (a, b) in changed.items())
            sys.exit(f"Fortsetzen nicht möglich: seit dem Start hat sich etwas geändert ({text}).\n"
                     f"Regel R8: bitte ein neues Training mit neuem Namen starten.")
        cp = run.find_checkpoint()
        if cp is None:
            sys.exit(f"Training '{args.name}' hat keinen Fortsetzungspunkt.")
    else:
        if run.exists or run.dir.exists():
            sys.exit(f"Training '{args.name}' gibt es schon. Zum Fortsetzen: --resume, sonst anderen Namen wählen.")
        reward_cfg = load_reward_config()
        cfg = new_config(args, train_cfg, reward_cfg)
        run.create(cfg)

    git = git_info()
    if git["geaendert"]:
        print("Hinweis: Es gibt nicht eingecheckte Änderungen am Code (werden in summary.csv vermerkt).")
    trainer = Trainer(run, cfg, reward_cfg, train_cfg["ressourcen"], train_cfg["profile"], args.profil,
                      max_sps=args.max_sps, stop_steps=args.schritte, stop_minutes=args.minuten)
    if args.resume:
        trainer.resume(cp)
        event = "fortsetzung"
        note = f"von {cp.name}, Commit {(git['commit'] or '?')[:7]}"
        if git["commit"] != cfg["git"]["commit"]:
            note += f" (Start war {(cfg['git']['commit'] or '?')[:7]})"
    else:
        trainer.new(args.seed)
        event, note = "start", f"Seed {args.seed}, Commit {(git['commit'] or '?')[:7]}"
    if git["geaendert"]:
        note += ", mit nicht eingecheckten Änderungen"
    trainer._log(event, bemerkung=note)

    caffeinate = keep_awake()
    v = cfg["versionen"]
    print(f"Training '{args.name}'  ({'fortgesetzt' if args.resume else 'neu'})  ->  {run.dir.relative_to(ROOT)}/")
    print(f"Versionen: Physik v{v['physik']}, Regeln v{v['regeln']}, Sensoren v{v['sensoren']}, "
          f"Strecke {cfg['strecke_id']} v{v['strecke']}, Reward v{v['reward']}")
    if caffeinate:
        print("Mac bleibt wach (caffeinate), solange das Training läuft. Deckel offen lassen.")
    print("Ctrl+C = pausieren und speichern\n")

    session = {"start": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": git["commit"], "profil": args.profil,
               "befehl": sys.argv}
    try:
        trainer.train(session)
    finally:
        if caffeinate:
            caffeinate.terminate()


if __name__ == "__main__":
    main()
