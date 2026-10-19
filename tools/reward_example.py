"""Reward an echten Zahlen erklären: Der Regel-Fahrer fährt eine Runde, ausgewählte Schritte werden aufgeschlüsselt.

    .venv/bin/python tools/reward_example.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rl.env import RaceEnv  # noqa: E402
from rl.rule_driver import rule_action  # noqa: E402

env = RaceEnv()
print(f"Gewichte: k_fortschritt={env.k_progress}, k_zeit={env.k_time}, k_crash={env.k_crash}, "
      f"Ziel-Bonus={'fest ' + str(env.k_finish) if env.k_finish_per_s is None else str(env.k_finish_per_s) + ' pro s unter dem Zeitlimit'}\n")
print(f"{'Schritt':>7} {'Zeit':>6} {'Tempo':>9} {'Höchststand':>12} {'+Fortschr.':>10} {'Zeit':>6} "
      f"{'Crash':>6} {'Ziel':>6} {'= Reward':>9} {'Summe':>8}")

obs, info = env.reset()
total, step, done = 0.0, 0, False
sums = {"fortschritt": 0.0, "zeit": 0.0, "crash": 0.0, "ziel": 0.0}
show = {1, 2, 60, 600, 1500}
while not done:
    obs, reward, terminated, truncated, info = env.step(rule_action(obs))
    step += 1
    total += reward
    for k, v in info["reward_teile"].items():
        sums[k] += v
    done = terminated or truncated
    if step in show or done:
        p = info["reward_teile"]
        print(f"{step:>7} {step / 60:>5.2f}s {obs[17] * 60 * 3.6:>6.1f}km/h {env.best_progress:>10.2f} m "
              f"{p['fortschritt']:>+10.3f} {p['zeit']:>+6.2f} {p['crash']:>+6.0f} {p['ziel']:>+6.0f} "
              f"{reward:>+9.3f} {total:>8.1f}")

print(f"\nEnde: {info['ende']}, Rundenzeit {info['rundenzeit']:.2f} s, {step} Schritte")
print("Zusammensetzung: " + ", ".join(f"{k} {v:+.1f}" for k, v in sums.items()) + f"  = {total:.1f}")

# Zum Vergleich: stehen bleiben
obs, info = env.reset()
total, done = 0.0, False
while not done:
    obs, reward, terminated, truncated, info = env.step(np.zeros(4))
    total += reward
    done = terminated or truncated
print(f"Zum Vergleich – stehen bleiben: Ende '{info['ende']}' nach {info['schritte']} Schritten, Reward {total:.1f}")
