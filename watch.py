"""Der KI beim Fahren zuschauen: lädt ein Modell und fährt deterministisch (ohne Zufall, R6) mit Grafik.

    .venv/bin/python watch.py test1 --live              # LIVE: nach jeder Runde das neueste Modell aus dem laufenden Training
    .venv/bin/python watch.py test1                     # offizielles Bestzeit-Modell (sonst letzter Stand)
    .venv/bin/python watch.py test1 --modell neuester   # jüngstes gespeichertes Modell (Version, Bestzeit, Fortsetzungspunkt)
    .venv/bin/python watch.py test1 --modell letzter    # letzter Fortsetzungspunkt
    .venv/bin/python watch.py test1 --modell runs/test1/versionen/model_0000250000.zip
    .venv/bin/python watch.py test1 --gegen-mich        # mit deiner Bestzeit als Geist

Tasten: R = nochmal, N/L = neueste Version sofort, Z = Zeitleiste ein/aus, C = Kamera, V = Sensorstrahlen,
        Esc = Beenden.
Zeitleiste (unten): mit der Maus ziehen oder ←/→ -> ältere Versionen ansehen; ganz rechts = LIVE.
Die KI sieht nur ihre Beobachtung (R5) und fährt mit derselben Simulation wie der Spieler (R3).
"""
import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("name", help="Name des Trainings (Ordner runs/<name>/)")
ap.add_argument("--modell", default="best", help="best, neuester, letzter oder Pfad zu einer .zip-Datei")
ap.add_argument("--live", action="store_true", help="nach jeder Runde das neueste Modell des laufenden Trainings laden")
ap.add_argument("--gegen-mich", action="store_true", help="eigene Bestzeit als Geist mitfahren lassen")
ap.add_argument("--ghost", action="append", default=[], help="weitere Geist-Datei (mehrfach möglich)")
args = ap.parse_args()

run_dir = ROOT / "runs" / args.name
if not (run_dir / "config.json").exists():
    sys.exit(f"Training '{args.name}' nicht gefunden (runs/{args.name}/config.json fehlt).")
cfg = json.loads((run_dir / "config.json").read_text())
cp = next((d for d in (run_dir / "checkpoint", run_dir / "checkpoint.alt") if (d / "model.zip").exists()), None)
live_file = run_dir / "live.json"
if args.live:
    model_path = run_dir / "live_model.zip"
    if not model_path.exists() or not live_file.exists():
        sys.exit("Noch kein Live-Modell. Es entsteht bei der nächsten Auswertung (alle 25'000 Schritte). Läuft das\n"
                 "Training noch mit altem Code: einmal Ctrl+C und mit --resume neu starten.")
elif args.modell == "best":
    model_path = run_dir / "best_model.zip" if (run_dir / "best_model.zip").exists() else cp and cp / "model.zip"
    if model_path and model_path.parent == cp:
        print("Noch keine offizielle Bestzeit – zeige den letzten Stand.")
elif args.modell == "neuester":
    candidates = [*(run_dir / "snapshots").glob("*.zip"), *(run_dir / "versionen").glob("*.zip"),
                  run_dir / "best_model.zip", *([cp / "model.zip"] if cp else [])]
    candidates = [p for p in candidates if p.exists()]
    model_path = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
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
live = json.loads(live_file.read_text()) if args.live else None
state = {}
state_file = (cp / "state.json") if cp else None
if state_file and state_file.exists():
    state = json.loads(state_file.read_text())

import render  # noqa: E402  GL-Setup für macOS, muss vor Ursina() geladen sein
from ursina import Ursina, application, camera, mouse, time, window  # noqa: E402

from game.car import DT  # noqa: E402
from game.ghost import Ghost, best_time  # noqa: E402
from render.cameras import CameraRig  # noqa: E402
from render.effects import DriftEffects, SpeedStreaks  # noqa: E402
from render.minimap import Minimap  # noqa: E402
from render.hud import HUD, format_time  # noqa: E402
from render.timeline import Timeline  # noqa: E402
from render.world import GhostCar, World, lerp_pose  # noqa: E402

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
effects = DriftEffects()
streaks = SpeedStreaks()

ghost_files = list(args.ghost)
mine = best_time(track.id)
if args.gegen_mich and mine:
    ghost_files.append(mine["ghost"])
ghosts = [GhostCar(Ghost(p), i) for i, p in enumerate(ghost_files)]
minimap = Minimap(track, [g.color for g in ghosts])

st = {"obs": None, "acc": 0.0, "prev": None, "message": "", "wait": 1.0, "end_wait": None, "scan": 0.0,
      "shown_steps": live["schritte"] if live else None}
timeline = Timeline()
versions_dir = run_dir / "versionen"


def scan_versions():
    """Alle gespeicherten Versionen {Schritte: Datei}. Im Live-Modus wird die aktuelle Live-Version
    mit aufbewahrt (falls das Training sie nicht selbst speichert)."""
    found = {}
    for folder in (run_dir / "snapshots", versions_dir):
        for f in folder.glob("model_*.zip"):
            try:
                found[int(f.stem.split("_")[1])] = f
            except (IndexError, ValueError):
                pass
    if args.live:
        try:
            info = json.loads(live_file.read_text())
            if info["schritte"] not in found and (run_dir / "live_model.zip").exists():
                versions_dir.mkdir(exist_ok=True)
                dst = versions_dir / f"model_{info['schritte']:010d}.zip"
                shutil.copyfile(run_dir / "live_model.zip", dst)
                found[info["schritte"]] = dst
        except (OSError, ValueError, KeyError):
            pass
    return dict(sorted(found.items()))


def eval_results():
    """Ergebnis jeder Auswertung aus evals.csv: {Schritte: Text}. Läufe von vor Schritt 5: summary.csv."""
    out = {}
    path = run_dir / "evals.csv"
    if not path.exists():
        path = run_dir / "summary.csv"
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("ereignis", "auswertung") != "auswertung":
                    continue
                if row["gueltig"] == "ja":
                    out[int(row["schritte"])] = f"Auswertung: Ziel in {format_time(float(row['rundenzeit']))}"
                else:
                    out[int(row["schritte"])] = f"Auswertung: {float(row['fortschritt_pct']):.0f} % ({row['ende']})"
    except (OSError, ValueError, KeyError):
        pass
    return out


versions = scan_versions()
results = eval_results()
timeline.set_versions(list(versions))


def load_version(steps):
    """Gewählte Version laden und die Fahrt neu starten."""
    global model
    path = versions.get(steps)
    if path is None:
        return
    model = SAC.load(path, device="cpu")
    st["shown_steps"] = steps
    restart()
LIVE_PAUSE = 2.0      # so lange bleibt das Ergebnis stehen, dann kommt der nächste Stand


def load_live():
    """Neuestes Modell des Trainings laden, falls es seit der letzten Runde ein neues gibt."""
    global model, live
    try:
        info = json.loads(live_file.read_text())
        if info["schritte"] != live["schritte"]:
            model = SAC.load(model_path, device="cpu")
            live = info
    except (OSError, ValueError, KeyError):
        pass          # Training schreibt gerade -> beim nächsten Mal


def pose():
    c = sim.car
    return (c.x, c.y, c.z, c.heading, c.pitch, c.roll, c.steer)


def restart():
    obs, _ = env.reset()
    for g in ghosts:
        g.reset()
    rig.first = True
    st.update(obs=obs, acc=0.0, prev=pose(), message="", wait=1.0, end_wait=None)


def on_end(info):
    if info["gueltige_runde"]:
        t = info["rundenzeit"]
        text = f"ZIEL!  {format_time(t)}"
        if mine:
            text += f"\nDu: {format_time(mine['time'])}  ({t - mine['time']:+.2f} s)"
    else:
        text = f"{info['ende'].upper()}\n{info['fortschritt_prozent']:.0f} % geschafft"
    if args.live and timeline.live:
        st["end_wait"] = LIVE_PAUSE
        st["message"] = text + "\ngleich: neuester Stand"
    else:
        st["message"] = text + "\n[R] nochmal   ·   Zeitleiste: andere Version"


def update():
    global versions, results
    st["scan"] -= time.dt
    if st["scan"] <= 0:                          # neue Versionen einsammeln
        st["scan"] = 2.0
        versions = scan_versions()
        results = eval_results()
        timeline.set_versions(list(versions))
    if timeline.dragging:
        timeline.drag_to_mouse()
    timeline.show_info(results.get(timeline.selected))
    if st["end_wait"] is not None:
        st["end_wait"] -= time.dt
        if st["end_wait"] <= 0:
            load_live()
            st["shown_steps"] = live["schritte"] if live else st["shown_steps"]
            restart()
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
    x, y, z, heading, pitch, roll, steer = lerp_pose(st["prev"], pose(), alpha)
    vf, _ = sim.car.local_velocity()
    world.car.set_state(x, y, z, heading, pitch, roll, steer)
    if world.rays.enabled:
        world.rays.set_state(x, y, z, heading)
    c = sim.car
    rig.update(x, y, z, heading, vf, time.dt, roll=roll, move_heading=math.atan2(c.vx, c.vz))
    world.follow(x, y, z)
    effects.update(c, time.dt)
    streaks.update(c, camera.world_position, time.dt)
    for g in ghosts:
        g.show_frame(sim.frame - 1 + alpha)
    minimap.update(x, z, heading, [(g.model.x, g.model.z) for g in ghosts])

    lines = [track.name]
    if live and st["shown_steps"] is not None and not (timeline.live and st["shown_steps"] == live["schritte"]):
        lines.append(f"Gezeigte Version: Schritt {st['shown_steps']:,}   (neueste: {live['schritte']:,})".replace(",", "'"))
        best = live.get("bestzeit")
        lines.append(f"Offizielle KI-Bestzeit: {format_time(best) if best is not None else '–'}   Stufe {live.get('stufe', 0)}")
    elif live:
        lines.append(f"LIVE · Stand: Schritt {live['schritte']:,} · Versuch {live['versuche']:,}".replace(",", "'"))
        best = live.get("bestzeit")
        lines.append(f"Offizielle KI-Bestzeit: {format_time(best) if best is not None else '–'}   Stufe {live.get('stufe', 0)}")
    else:
        lines.append(f"Modell: {model_label}")
    if state and not live:
        best = state.get("bestzeit")
        lines.append(f"Offizielle KI-Bestzeit: {format_time(best) if best is not None else '–'}"
                     f"   ({state.get('schritte', 0):,} Schritte, {state.get('versuche', 0):,} Versuche)".replace(",", "'"))
    lines.append(f"Deine Bestzeit: {format_time(mine['time'])}" if mine else "Deine Bestzeit: –")
    if other:
        lines.append(f"Achtung: andere Version als im Training ({', '.join(other)}) – Zeiten nicht vergleichbar")
    lines.append(f"Wandberührungen: {sim.wall_hits}")
    lines.append(f"Kamera: {rig.mode} [C]   Sensoren [V]   Nochmal [R]   Zeitleiste [Z]"
                 + ("   Neueste Version [L]" if args.live else ""))
    hud.set(vf, sim.time, "\n".join(lines), st["message"])
    n_cp = len(track.checkpoints)
    best = (live or state or {}).get("bestzeit")
    hud.set_bar(f"{min(len(sim.splits), n_cp)}/{n_cp}", format_time(best) if best is not None else "–")
    if st["message"]:
        hud.big.set(st["message"], HUD.YELLOW)


def input(key):
    if key in ("r", "backspace"):
        restart()
    elif key in ("n", "l") and args.live:          # sofort die neueste Version laden und neu starten
        timeline.go_live()
        load_live()
        st["shown_steps"] = live["schritte"]
        restart()
    elif key in ("left arrow", "right arrow"):     # Zeitleiste: eine Version zurück/vor
        timeline.step(-1 if key == "left arrow" else 1)
        _pick()
    elif key == "left mouse down" and timeline.hovered():
        timeline.dragging = True
    elif key == "left mouse up" and timeline.dragging:
        timeline.dragging = False
        _pick()
    elif key == "z":                              # Zeitleiste ein-/ausblenden
        timeline.toggle()
    elif key == "c":
        rig.next_mode()
    elif key == "v":
        world.rays.enabled = not world.rays.enabled
    elif key == "escape":
        application.quit()


def _pick():
    """Version aus der Zeitleiste laden (live = immer die neueste)."""
    if timeline.live and args.live:
        load_live()
        st["shown_steps"] = live["schritte"]
        restart()
    elif timeline.selected is not None and timeline.selected != st["shown_steps"]:
        load_version(timeline.selected)


restart()
app.run()
