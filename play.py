"""Selbst fahren (Spiel v2, 3D) – optional gegen Geister.

    python play.py                                    # Strecke 1
    python play.py tracks/poly02.json --ghost ghosts/<datei>.json [--ghost ...]
    python play.py --best                             # gegen die eigene Bestzeit fahren

Tastatur: W/S oder Pfeile = Gas/Bremse, A/D = Lenken, Leertaste = Drift, T/R/Backspace = Neustart,
          C = Kamera, V = KI-Sensoren, Tab oder F3 = Debug-Anzeige, Esc = Beenden
Gamepad:  linker Stick = Lenken (stufenlos), RT = Gas, LT = Bremse, X/RB = Drift, Y/Start = Neustart

Fertige Fahrten (und Crashs ab 5 % Fortschritt) werden als Geist in ghosts/ gespeichert,
die Bestzeit pro Strecke in saves/bestzeiten.json.
"""
import argparse
import gc
import math

import render  # GL-Setup für macOS, muss vor Ursina() geladen sein
from ursina import Ursina, application, camera, time, window

from game.car import DT
from game.ghost import Ghost, best_time, save_ghost, submit_time
from game.sim import END_REASONS, PHYSICS_VERSION, RULES_VERSION, SENSOR_VERSION, Sim
from game.track import Track
from render.cameras import CameraRig
from render.effects import DriftEffects, SpeedStreaks
from render.controls import RESTART_KEYS, Controls
from render.hud import HUD, TextLines, format_delta, format_time
from render.minimap import Minimap
from render.world import GhostCar, World, lerp_pose

ap = argparse.ArgumentParser()
ap.add_argument("track", nargs="?", default="tracks/poly01.json")
ap.add_argument("--ghost", action="append", default=[], help="Geist-Datei (mehrfach möglich)")
ap.add_argument("--best", action="store_true", help="eigene Bestzeit als Geist mitfahren lassen")
args = ap.parse_args()

track = Track.load(args.track)
sim = Sim(track)
controls = Controls()

app = Ursina(title=f"KI-Fahrschule – {track.name}", size=(1280, 720), development_mode=False)
render.fix_after_start()
window.fps_counter.enabled = False
window.exit_button.visible = False
world = World(track)
hud = HUD()
rig = CameraRig(track, mode="chase")
effects = DriftEffects()
streaks = SpeedStreaks()

ghost_files = list(args.ghost)
record = best_time(track.id)
if args.best and record:
    ghost_files.append(record["ghost"])
ghosts = [GhostCar(Ghost(p), i) for i, p in enumerate(ghost_files)]
minimap = Minimap(track, [g.color for g in ghosts])
debug = TextLines(position=(-0.86, 0.04), height=0.03)
debug.enabled = False
reference = ghosts[0].ghost if ghosts else (Ghost(record["ghost"]) if record else None)

COUNTDOWN_FIRST, COUNTDOWN_RESTART = 3.0, 1.5
state = {"phase": "countdown", "countdown": COUNTDOWN_FIRST, "countdown_total": COUNTDOWN_FIRST, "acc": 0.0,
         "message": "", "split_text": "", "split_color": HUD.WHITE, "split_until": 0.0}


def car_pose():
    c = sim.car
    return (c.x, c.y, c.z, c.heading, c.pitch, c.roll, c.steer)


state["prev_pose"] = car_pose()


def restart(countdown=COUNTDOWN_RESTART):
    sim.reset()
    controls.reset()
    for g in ghosts:
        g.reset()
    effects.reset()
    hud.set_splits()
    rig.first = True
    state.update(phase="countdown", countdown=countdown, countdown_total=countdown, acc=0.0, message="",
                 split_text="", split_until=0.0, wall_until=0.0, prev_pose=car_pose(), delta=("–", HUD.WHITE))


def on_checkpoint(name, split_time, index):
    text, col = f"{name}   {format_time(split_time)}", HUD.WHITE
    ref = reference.split(index) if reference else None
    best = speed = ""
    speed_col = HUD.WHITE
    if ref is not None:
        delta = split_time - ref
        text += f"   {format_delta(delta)}"
        col = HUD.GREEN if delta < -0.005 else HUD.RED if delta > 0.005 else HUD.WHITE
        state["delta"] = (format_delta(delta), col)
        best = format_time(ref)
        ref_frame = int(round(ref / DT))                       # Tempo des Rekords an derselben Stelle
        dv = (sim.car.speed - reference.speed_at(ref_frame)) * 3.6
        speed, speed_col = f"{dv:+.0f} km/h", (HUD.GREEN if dv >= 0 else HUD.RED)
    hud.set_splits(best, format_time(split_time), state["delta"][0] if ref is not None else "", col, speed, speed_col)
    state.update(split_text="", split_color=col, split_until=sim.time + 3.0)


def on_end():
    state["phase"] = "ended"
    if sim.result == "finished":
        path = save_ghost(sim, "player")
        new_record = submit_time(track.id, sim.finish_time, path)
        state["message"] = f"ZIEL!  {format_time(sim.finish_time)}" + ("\nNEUER REKORD!" if new_record else "")
    else:
        if sim.progress_pct >= 5:
            save_ghost(sim, "player")
        title = "ZEITLIMIT" if sim.result == "truncated" else f"CRASH – {END_REASONS[sim.end_reason]}"
        state["message"] = f"{title}\n{sim.progress_pct:.0f} % geschafft"
    state["message"] += "\n[R] nochmal"


def update():
    if state["phase"] == "countdown":
        state["countdown"] -= time.dt
        if state["countdown"] <= 0:
            state["phase"] = "racing"
    elif state["phase"] == "racing":
        state["acc"] += min(time.dt, 0.1)
        while state["acc"] >= DT and not sim.done:  # feste 60 Hz Physik, unabhängig von FPS
            state["acc"] -= DT
            state["prev_pose"] = car_pose()
            events = sim.step(controls.read(DT))
            if events["wall_hit"]:
                state["wall_until"] = sim.time + 0.6
            if events["checkpoint"]:
                on_checkpoint(events["checkpoint"], events["split"], len(sim.splits) - 1)
        if sim.done:
            on_end()

    # ---- Anzeige: zwischen den letzten beiden Physik-Schritten überblenden (kein Ruckeln) ----
    alpha = state["acc"] / DT if state["phase"] == "racing" and not sim.done else 1.0
    x, y, z, heading, pitch, roll, steer = lerp_pose(state["prev_pose"], car_pose(), alpha)
    c = sim.car
    vf, _ = c.local_velocity()
    world.car.set_state(x, y, z, heading, pitch, roll, steer)
    if world.rays.enabled:
        world.rays.set_state(x, y, z, heading)
    rig.update(x, y, z, heading, vf, time.dt, roll=roll, move_heading=math.atan2(c.vx, c.vz))
    world.follow(x, y, z)
    effects.update(c, time.dt)
    streaks.update(c, camera.world_position, time.dt)
    for g in ghosts:
        g.show_frame(sim.frame - 1 + alpha)
    minimap.update(x, z, heading, [(g.model.x, g.model.z) for g in ghosts])

    n_cp = len(track.checkpoints)
    info = f"{track.name}" + ("" if record else "\nnoch keine Bestzeit")
    for g in ghosts:
        info += f"\nGeist: {g.ghost.label} · {g.ghost.result_text}"
    info += f"\nKamera [C]   Sensoren [V]   Debug [Tab]   Neustart [T]"
    passed = len(sim.splits) if not sim.done else n_cp
    hud.set_bar(f"{min(passed, n_cp)}/{n_cp}", format_time(record["time"]) if record else "–",
                *state.get("delta", ("–", HUD.WHITE)))

    big = state["message"]
    if state["phase"] == "countdown":
        big = str(math.ceil(state["countdown"] / (state["countdown_total"] / 3)))
    elif state["phase"] == "racing" and sim.time < 0.8:
        big = "GO!"
    hud.set(vf, sim.time, info, big)
    vf_, vr_ = c.local_velocity()
    drift_angle = abs(math.degrees(math.atan2(vr_, vf_))) if vf_ > 5 else 0.0
    hud.drift.text = f"DRIFT {drift_angle:.0f}°" if c.drift > 0.3 and drift_angle > 5 else ""
    if big:
        hud.big.set(big, HUD.YELLOW)
    elif sim.time < state.get("wall_until", 0):
        hud.big.set("AUTSCH!", HUD.RED)
    if debug.enabled:
        show_debug(vf)
    if sim.time < state["split_until"]:
        hud.split.set(state["split_text"], state["split_color"])
    else:
        hud.split.text = ""
        hud.set_splits()


gc.collect()
gc.freeze()


def show_debug(speed):
    n_cp = len(track.checkpoints)
    c = sim.car
    gate = sim.gates[sim.next_gate].name if not sim.done else "–"
    debug.text = "\n".join([
        "DEBUG",
        f"Fortschritt: {sim.progress_pct:5.1f} %  ({sim.progress:.0f} von {track.length:.0f} m)",
        f"Tempo: {speed * 3.6:5.1f} km/h   senkrecht {c.vy:+5.1f} m/s",
        f"Höhe: {c.y:6.1f} m   Neigung {math.degrees(c.pitch):+5.1f}°   {'IN DER LUFT' if not c.grounded else 'am Boden'}",
        f"Checkpoint: {min(sim.next_gate, n_cp)}/{n_cp} geschafft, nächstes Tor: {gate}",
        f"Zeit: {format_time(sim.time)}  ({sim.frame} Physik-Schritte)",
        f"Versionen: Physik v{PHYSICS_VERSION}, Regeln v{RULES_VERSION}, "
        f"Sensoren v{SENSOR_VERSION}, Strecke {track.id} v{track.version}",
    ])


def input(key):
    global record
    if key in RESTART_KEYS:
        record = best_time(track.id)
        restart()
    elif key == "c":
        rig.next_mode()
    elif key in ("tab", "f3"):
        debug.enabled = not debug.enabled
    elif key == "v":
        world.rays.enabled = not world.rays.enabled
    elif key == "escape":
        application.quit()


app.run()
