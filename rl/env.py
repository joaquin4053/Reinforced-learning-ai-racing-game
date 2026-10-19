"""Gymnasium-Umgebung für das KI-Training (Plan-Schritt 3).

    from rl.env import RaceEnv
    env = RaceEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

- Beobachtung: der Vektor aus game/sensors.py (Tabelle in CLAUDE.md), Werte -1..1 bzw. 0..1
- Aktion: [Lenkung -1..1, Gas 0..1, Bremse 0..1, Drift 0..1]  (Drift abschaltbar, R4)
- Reward: genau nach Regel R7, Gewichte in reward_config.toml
- Ende: Ziel oder Crash -> terminated, Zeitlimit -> truncated (R11)
"""
import tomllib
from pathlib import Path

import gymnasium as gym
import numpy as np

from game import sensors
from game.sim import END_REASONS, PHYSICS_VERSION, RULES_VERSION, SENSOR_VERSION, Sim
from game.track import Track

ROOT = Path(__file__).resolve().parent.parent
REWARD_CONFIG = ROOT / "reward_config.toml"

END_TEXT = {"finished": "Ziel", "fell": "Runtergefallen", "stuck": "Stillstand", "wrong_way": "Falsche Richtung",
            "time_limit": "Zeitlimit"}


def load_reward_config(path=REWARD_CONFIG):
    with open(path, "rb") as f:
        return tomllib.load(f)


class RaceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, track_file=None, config=None, record=False):
        self.config = config or load_reward_config()
        w = self.config["gewichte"]
        self.k_progress, self.k_time = w["k_fortschritt"], w["k_zeit"]
        self.k_crash, self.k_finish = w["k_crash"], w["k_ziel"]
        env_cfg = self.config["umgebung"]
        self.track = Track.load(ROOT / (track_file or env_cfg["strecke"]))
        self.sim = Sim(self.track, record=record)
        self.drift = env_cfg["drift_erlaubt"]

        self.observation_space = gym.spaces.Box(sensors.LOW, sensors.HIGH, (sensors.SIZE,), np.float32)
        low, high = [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]           # Lenkung, Gas, Bremse (R4)
        if self.drift:
            low, high = low + [0.0], high + [1.0]                # + Drift
        self.action_space = gym.spaces.Box(np.array(low, np.float32), np.array(high, np.float32))
        self.versions = {"physik": PHYSICS_VERSION, "regeln": RULES_VERSION, "sensoren": SENSOR_VERSION,
                         "strecke": self.track.version, "reward": self.config["version"]["reward"]}

    # ---------- Gymnasium-Schnittstelle ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)   # die Simulation selbst hat keinen Zufall (R6: deterministisch)
        obs = self.sim.reset()
        self.best_progress = self.sim.progress   # Höchststand des Fortschritts in Metern
        self.episode_reward = 0.0
        self.last_reward = {}
        return obs, self._info()

    def step(self, action):
        act = np.asarray(action, dtype=np.float64).reshape(-1)
        if not self.drift:
            act = np.append(act[:3], 0.0)
        self.sim.step(act)    # zu große Werte schneidet die Simulation ab (R4)

        # --- Reward nach R7 ---
        new_best = max(self.best_progress, self.sim.progress)
        parts = {
            "fortschritt": self.k_progress * (new_best - self.best_progress),   # nur neuer Höchststand
            "zeit": -self.k_time,
            "crash": -self.k_crash if self.sim.result == "crashed" else 0.0,
            "ziel": self.k_finish if self.sim.result == "finished" else 0.0,
        }
        self.best_progress = new_best
        reward = float(sum(parts.values()))
        self.last_reward = parts
        self.episode_reward += reward

        terminated = self.sim.result in ("finished", "crashed")
        truncated = self.sim.result == "truncated"
        return self.sim.observation(), reward, terminated, truncated, self._info()

    # ---------- Zusatzinfos ----------
    def _info(self):
        sim = self.sim
        reason = None
        if sim.result == "finished":
            reason = "Ziel"
        elif sim.result is not None:
            reason = END_TEXT.get(sim.end_reason, END_REASONS.get(sim.end_reason, sim.end_reason))
        return {
            "gueltige_runde": sim.result == "finished",            # R1
            "rundenzeit": sim.finish_time,                          # R6, nur bei gültiger Runde
            "fortschritt_prozent": sim.progress_pct,
            "ende": reason,                                         # Ziel, Stillstand, Falsche Richtung, Zeitlimit
            "crash": sim.result == "crashed",
            "wandberuehrungen": sim.wall_hits,
            "schritte": sim.frame,
            "reward_teile": dict(self.last_reward),
            "versionen": dict(self.versions),
        }
