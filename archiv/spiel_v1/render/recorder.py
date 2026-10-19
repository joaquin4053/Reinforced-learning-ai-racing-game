"""Einen Geist (gespeicherte Fahrt) mit der Kinokamera abspielen und als MP4 aufnehmen.

Ursina kann pro Prozess nur einmal gestartet werden -> immer in eigenem Prozess aufrufen
(replay.py). Das Training startet replay.py später alle 100 Versuche als Subprozess.
"""
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from ursina import Ursina, window

import render
from game.ghost import Ghost
from game.sim import END_REASONS
from game.track import Track
from render.cameras import CameraRig
from render.minimap import Minimap
from render.world import HUD, World, format_time


def _grab(app):
    tex = app.win.getScreenshot()
    w, h = tex.getXSize(), tex.getYSize()
    img = np.frombuffer(tex.getRamImageAs("RGB").getData(), np.uint8).reshape(h, w, 3)
    return img[::-1]  # Panda3D liefert das Bild auf dem Kopf


def _progress_per_frame(track, frames):
    """Fortschritt (Meter) für jede Position – für die TV-Kamera und die Anzeige."""
    out, seg, raw, total = [], None, None, 0.0
    for x, z, _ in frames:
        dist, seg, _, _ = track.progress_at(x, z, hint=seg)
        if raw is not None:
            d = dist - raw
            if track.is_loop and abs(d) > track.length / 2:
                d -= track.length if d > 0 else -track.length
            total += d
        else:
            total = dist
        raw = dist
        out.append(total)
    return out


def render_video(ghost_path, out_path=None, label=None, show_rays=True, size=(1280, 720),
                 seed=0, offscreen=True, hold_seconds=2.5, max_seconds=None):
    ghost = Ghost(ghost_path)
    meta = ghost.meta
    track = Track.load(Path("tracks") / f"{meta['map']}.json")
    frames = ghost.frames
    dt = meta["dt"]
    fps = round(1 / dt)
    if max_seconds:
        frames = frames[:int(max_seconds / dt)]
    if out_path is None:
        Path("videos").mkdir(exist_ok=True)
        out_path = f"videos/{Path(ghost_path).stem}.mp4"
    label = label if label is not None else ghost.label

    app = Ursina(title="KI-Fahrschule Replay", size=size, development_mode=False,
                 window_type="offscreen" if offscreen else "onscreen")
    render.fix_after_start()
    if not offscreen:
        window.fps_counter.enabled = False
        window.exit_button.visible = False
    world = World(track)
    world.rays.enabled = show_rays
    hud = HUD(label)
    rig = CameraRig(track, mode="cinematic", seed=seed)
    minimap = Minimap(track)
    progress = _progress_per_frame(track, frames)
    splits = meta["splits"]
    gate_names = [g.name for g in track.checkpoints] + ["Ziel"]

    last = len(frames) - 1
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=5,
                                macro_block_size=2, ffmpeg_log_level="error")
    for _ in range(3):  # ein paar Frames zum Aufwärmen (Shader, Schatten)
        app.step()
    for i in range(last + 1 + int(hold_seconds * fps)):
        f = min(i, last)
        x, z, h = frames[f]
        t = f * dt
        world.car.set_state(x, z, h)
        if show_rays:
            world.rays.set_state(x, z, h)
        speed = ghost.speed_at(f) if f > 0 else 0.0
        rig.update(x, z, h, speed, progress[f], dt)
        minimap.update(x, z, h)

        big = ""
        if f == last and i >= last:
            if meta["result"] == "finished":
                big = f"ZIEL!  {format_time(meta['finish_time'])}"
            else:
                title = "ZEITLIMIT" if meta["result"] == "truncated" else f"CRASH – {END_REASONS.get(meta['end_reason'], '')}"
                big = f"{title}\n{meta['progress_pct']:.0f} % geschafft"
        pct = 100 * progress[f] / (track.length * track.laps)
        hud.set(speed, t, f"{track.name}\nStrecke: {max(0.0, min(pct, 100.0)):4.1f} %", big)
        # Zwischenzeit 2.5 s lang einblenden
        recent = [(k, s) for k, s in enumerate(splits) if s <= t < s + 2.5]
        if recent:
            k, s = recent[-1]
            hud.split.text = f"{gate_names[k % len(gate_names)]}   {format_time(s)}"
        else:
            hud.split.text = ""
        app.step()
        writer.append_data(_grab(app))
    writer.close()
    return out_path
