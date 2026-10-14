"""Der KI beim Fahren zuschauen: lädt ein Modell und fährt deterministisch (ohne Zufall, R6) mit Grafik.

    .venv/bin/python watch.py test1                     # offizielles Bestzeit-Modell (sonst letzter Stand)
    .venv/bin/python watch.py test1 --modell letzter    # letzter Fortsetzungspunkt
    .venv/bin/python watch.py test1 --modell runs/test1/snapshots/model_0000250000.zip
    .venv/bin/python watch.py test1 --gegen-mich        # mit deiner Bestzeit als Geist

Tasten: R = nochmal, C = Kamera, V = Sensorstrahlen, Esc = Beenden.
Die KI sieht nur ihre Beobachtung (R5) und fährt mit derselben Simulation wie der Spieler (R3).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("name", help="Name des Trainings (Ordner runs/<name>/)")
ap.add_argument("--modell", default="best", help="best, letzter oder Pfad zu einer .zip-Datei")
ap.add_argument("--gegen-mich", action="store_true", help="eigene Bestzeit als Geist mitfahren lassen")
ap.add_argument("--ghost", action="append", default=[], help="weitere Geist-Datei (mehrfach möglich)")
args = ap.parse_args()

run_dir = ROOT / "runs" / args.name
if not (run_dir / "config.json").exists():
    sys.exit(f"Training '{args.name}' nicht gefunden (runs/{args.name}/config.json fehlt).")
cfg = json.loads((run_dir / "config.json").read_text())
cp = next((d for d in (run_dir / "checkpoint", run_dir / "checkpoint.alt") if (d / "model.zip").exists()), None)
if args.modell == "best":
    model_path = run_dir / "best_model.zip" if (run_dir / "best_model.zip").exists() else cp and cp / "model.zip"
    if model_path and model_path.parent == cp:
        print("Noch keine offizielle Bestzeit – zeige den letzten Stand.")
elif args.modell == "letzter":
    model_path = cp and cp / "model.zip"
else:
    model_path = Path(args.modell)
if not model_path or not Path(model_path).exists():
    sys.exit(f"Kein Modell gefunden ({args.modell}).")

import torch  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402

from rl.env import RaceEnv  # noqa: E402

torch.set_num_threads(1)
env = RaceEnv(config=cfg["reward_config"], record=False)
if env.versions["sensoren"] != cfg["versionen"]["sensoren"]:
    sys.exit("Die Sensoren haben sich seit dem Training geändert – dieses Modell passt nicht mehr (R8).")
other = [k for k, v in env.versions.items() if cfg["versionen"].get(k) != v]
model = SAC.load(model_path, device="cpu")
state = {}
state_file = (cp / "state.json") if cp else None
if state_file and state_file.exists():
    state = json.loads(state_file.read_text())

import render  # noqa: E402  GL-Setup für macOS, muss vor Ursina() geladen sein
from ursina import Ursina, application, time, window  # noqa: E402

from game.car import DT  # noqa: E402
from game.ghost import Ghost, best_time  # noqa: E402
from render.cameras import CameraRig  # noqa: E402
from render.minimap import Minimap  # noqa: E402
from render.world import HUD, GhostCar, World, format_time, lerp_pose  # noqa: E402

sim, track = env.sim, env.track
model_label = Path(model_path).resolve().relative_to(ROOT) if Path(model_path).resolve().is_relative_to(ROOT) \
    else model_path
app = Ursina(title=f"KI-Fahrschule – KI '{args.name}' – {track.name}", size=(1280, 720), development_mode=False)
render.fix_after_start()
window.fps_counter.enabled = False
window.exit_button.visible = False
world = World(track)
hud = HUD(label=f"KI · {args.name}")
rig = CameraRig(track, mode="chase")

ghost_files = list(args.ghost)
mine = best_time(track.id)
if args.gegen_mich and mine:
    ghost_files.append(mine["ghost"])
ghosts = [GhostCar(Ghost(p), i) for i, p in enumerate(ghost_files)]
minimap = Minimap(track, [g.color for g in ghosts])

st = {"obs": None, "acc": 0.0, "prev": None, "message": "", "wait": 1.0}


def pose():
    c = sim.car
    return (c.x, c.z, c.heading, c.steer)


def restart():
    obs, _ = env.reset()
    for g in ghosts:
        g.reset()
    rig.first = True
    st.update(obs=obs, acc=0.0, prev=pose(), message="", wait=1.0)


def on_end(info):
    if info["gueltige_runde"]:
        t = info["rundenzeit"]
        text = f"ZIEL!  {format_time(t)}"
        if mine:
            text += f"\nDu: {format_time(mine['time'])}  ({t - mine['time']:+.2f} s)"
    else:
        text = f"{info['ende'].upper()}\n{info['fortschritt_prozent']:.0f} % geschafft"
    st["message"] = text + "\n[R] nochmal"


def update():
    if st["wait"] > 0:                       # kurz stehen, damit man den Start sieht
        st["wait"] -= time.dt
    elif not sim.done:
        st["acc"] += min(time.dt, 0.1)
        while st["acc"] >= DT and not sim.done:   # feste 60 Hz Physik, eine Entscheidung pro Schritt (R4)
            st["acc"] -= DT
            st["prev"] = pose()
            act, _ = model.predict(st["obs"], deterministic=True)
            st["obs"], _, terminated, truncated, info = env.step(act)
            if terminated or truncated:
                on_end(info)

    alpha = st["acc"] / DT if not sim.done and st["wait"] <= 0 else 1.0
    x, z, heading, steer = lerp_pose(st["prev"], pose(), alpha)
    vf, _ = sim.car.local_velocity()
    world.car.set_state(x, z, heading, steer)
    if world.rays.enabled:
        world.rays.set_state(x, z, heading)
    rig.update(x, z, heading, vf, sim.progress, time.dt)
    for g in ghosts:
        g.show_frame(sim.frame - 1 + alpha)
    minimap.update(x, z, heading, [(g.model.x, g.model.z) for g in ghosts])

    lines = [track.name, f"Modell: {model_label}"]
    if state:
        best = state.get("bestzeit")
        lines.append(f"Offizielle KI-Bestzeit: {format_time(best) if best is not None else '–'}"
                     f"   ({state.get('schritte', 0):,} Schritte, {state.get('versuche', 0):,} Versuche)".replace(",", "'"))
    lines.append(f"Deine Bestzeit: {format_time(mine['time'])}" if mine else "Deine Bestzeit: –")
    if other:
        lines.append(f"Achtung: andere Version als im Training ({', '.join(other)}) – Zeiten nicht vergleichbar")
    lines.append(f"Wandberührungen: {sim.wall_hits}")
    lines.append(f"Kamera: {rig.mode} [C]   Sensoren [V]   Nochmal [R]")
    hud.set(vf, sim.time, "\n".join(lines), st["message"])
    if st["message"]:
        hud.big.set(st["message"], HUD.YELLOW)


def input(key):
    if key in ("r", "backspace"):
        restart()
    elif key == "c":
        rig.next_mode()
    elif key == "v":
        world.rays.enabled = not world.rays.enabled
    elif key == "escape":
        application.quit()


restart()
app.run()
