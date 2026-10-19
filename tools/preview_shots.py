"""Vorschau-Bilder einer Strecke ohne Fenster rendern (Testfahrer fährt, an ausgewählten Stellen ein Foto).

    .venv/bin/python tools/preview_shots.py tracks/poly01.json ausgabe_ordner [--modus chase]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import render  # noqa: E402
from ursina import Ursina  # noqa: E402

from game.sim import Sim  # noqa: E402
from game.track import Track  # noqa: E402
from render.cameras import CameraRig  # noqa: E402
from render.effects import SpeedStreaks  # noqa: E402
from render.hud import HUD  # noqa: E402
from render.minimap import Minimap  # noqa: E402
from render.world import World  # noqa: E402
from tools.autopilot import autopilot  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("track")
ap.add_argument("out")
ap.add_argument("--modus", default="chase")
ap.add_argument("--speed", type=float, default=45)
ap.add_argument("--ohne-schatten", action="store_true")
args = ap.parse_args()

track = Track.load(args.track)
sim = Sim(track)
poses, events = [], []
while not sim.done:
    ev = sim.step(autopilot(sim, args.speed))
    c = sim.car
    poses.append((c.x, c.y, c.z, c.heading, c.pitch, c.roll, c.steer, c.local_velocity()[0], sim.time,
                  sim.progress, c.grounded, math.atan2(c.vx, c.vz)))
# Fotos: kurz nach dem Start, mitten in Kurven/Steigungen (gleichmässig verteilt) und im Flug
airborne = [i for i, p in enumerate(poses) if not p[10]]
picks = [90] + [int(len(poses) * f) for f in (0.2, 0.4, 0.6, 0.8)]
if airborne:
    picks.append(airborne[len(airborne) // 2])

app = Ursina(title="Vorschau", size=(1280, 720), development_mode=False, window_type="offscreen")
render.fix_after_start()
world = World(track, shadows=not args.ohne_schatten)
hud = HUD()
rig = CameraRig(track, mode=args.modus)
minimap = Minimap(track)
streaks = SpeedStreaks()
from ursina import camera  # noqa: E402
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
for n, i in enumerate(sorted(picks)):
    rig.first = True
    for k in range(max(0, i - 90), i + 1):                # Kamera einschwingen lassen
        x, y, z, h, pi, ro, st, vf, t, prog, gr, mh = poses[k]
        world.car.set_state(x, y, z, h, pi, ro, st)
        rig.update(x, y, z, h, vf, 1 / 60, roll=ro, move_heading=mh)
        world.follow(x, y, z)
        sim.car.x, sim.car.y, sim.car.z, sim.car.heading = x, y, z, h
        sim.car.vx, sim.car.vz = vf * math.sin(mh), vf * math.cos(mh)
        streaks.update(sim.car, camera.world_position, 1 / 60)
    minimap.update(x, z, h)
    hud.set(vf, t, f"{track.name}\n{prog:.0f} m", "")
    hud.set_bar("1/3", "0:34.12", "+0.40 s", HUD.RED)
    for _ in range(3):
        app.step()
    tex = app.win.getScreenshot()
    w, hh = tex.getXSize(), tex.getYSize()
    img = np.frombuffer(tex.getRamImageAs("RGB").getData(), np.uint8).reshape(hh, w, 3)[::-1]
    name = out / f"{track.id}_{n}_{int(prog)}m{'_luft' if not gr else ''}.png"
    Image.fromarray(img).save(name)
    print(name)
