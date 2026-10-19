"""Aufzeichnung während des Trainings (Plan-Schritt 5): jeder Versuch als Datei, schlanke Modell-Snapshots.

episodes/versuch_0000123.npz   ein Versuch (komprimiert), lesen mit load_episode()
    meta         JSON: Versuchsnummer, Art (training/auswertung), Reward, Ergebnis, Rundenzeit, Dauer, Versionen, ...
    pos          (n+1, 3) float32  x, y, z in m – Zustand vor jedem Schritt und nach dem letzten
    rot          (n+1, 3) float16  Richtung (heading), Neigung (pitch), Querneigung (roll) in rad
    obs          (n+1, 24) float16 Beobachtung der KI (Tabelle in CLAUDE.md); obs[i] = Grundlage für aktion[i]
    aktion       (n, 4) float16    Lenkung, Gas, Bremse, Drift (abgeschnitten wie in der Simulation, R4)
    reward       (n,) float16      Reward jedes Schritts (exakte Summe steht in meta["reward"])
  Die Position ist float32, alles andere float16: Die Strecken sind über 1 km gross, float16 hätte dort
  nur noch 1 m Auflösung (das Auto würde in der Wiedergabe springen).

torch wird erst beim Speichern/Laden eines Snapshots geladen (viewer.py braucht es nicht).

models/versuch_0000120.pt      Gewichte von Actor und Critic (ohne Replay-Buffer, Optimizer, Ziel-Netze),
                               lesen mit load_snapshot()
"""
import io
import json
import zipfile

import numpy as np

from game import sensors

EPISODE_FORMAT = 1
SNAPSHOT_FORMAT = 1
ACTION_LOW = np.array([-1.0, 0.0, 0.0, 0.0])        # Lenkung, Gas, Bremse, Drift (R4)
ACTION_HIGH = np.array([1.0, 1.0, 1.0, 1.0])


class EpisodeRecorder:
    """Sammelt in der Umgebung jeden Schritt einer Fahrt (läuft auch in den Unterprozessen)."""

    def start(self, sim, obs):
        self.pos, self.rot, self.obs = [], [], []
        self.actions, self.rewards = [], []
        self._state(sim, obs)

    def add(self, action, reward, sim, obs):
        self.actions.append(action)
        self.rewards.append(reward)
        self._state(sim, obs)

    def _state(self, sim, obs):
        c = sim.car
        self.pos.append((c.x, c.y, c.z))
        self.rot.append((c.heading, c.pitch, c.roll))
        self.obs.append(obs)

    def finish(self):
        return {"pos": np.asarray(self.pos, np.float32), "rot": np.asarray(self.rot, np.float16),
                "obs": np.asarray(self.obs, np.float16),
                "aktion": np.asarray(self.actions, np.float16).reshape(-1, 4),
                "reward": np.asarray(self.rewards, np.float16)}


# ---------- Episoden-Dateien ----------
def episode_name(attempt):
    return f"versuch_{attempt:07d}.npz"


def save_episode(folder, attempt, arrays, meta):
    folder.mkdir(exist_ok=True)
    meta = {"format": EPISODE_FORMAT, **meta}
    path = folder / episode_name(attempt)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, meta=np.frombuffer(json.dumps(meta, ensure_ascii=False).encode(), np.uint8), **arrays)
    tmp.replace(path)
    return path


def load_episode(path):
    """-> (meta, arrays). Werte als float32, damit man direkt damit rechnen kann."""
    with np.load(path) as z:
        meta = json.loads(z["meta"].tobytes().decode())
        arrays = {k: z[k].astype(np.float32) for k in z.files if k != "meta"}
    return meta, arrays


def read_episode_meta(path):
    """Nur die Metadaten (schnell, ohne die Schritte zu entpacken)."""
    with zipfile.ZipFile(path) as zf, zf.open("meta.npy") as f:
        return json.loads(np.lib.format.read_array(io.BytesIO(f.read())).tobytes().decode())


# ---------- Modell-Snapshots ----------
def snapshot_name(attempt):
    return f"versuch_{attempt:07d}.pt"


def save_snapshot(folder, attempt, model, meta):
    """Nur Actor und Critic. Genug für Netz-Grafik, Critic-Kurve und Geister, nicht zum Weitertrainieren."""
    import torch
    folder.mkdir(exist_ok=True)
    pol = model.policy
    data = {"format": SNAPSHOT_FORMAT,
            "actor": {k: v.detach().cpu().clone() for k, v in pol.actor.state_dict().items()},
            "critic": {k: v.detach().cpu().clone() for k, v in pol.critic.state_dict().items()},
            "meta": {**meta, "netz": list(pol.net_arch) if isinstance(pol.net_arch, list) else pol.net_arch,
                     "obs_dim": int(model.observation_space.shape[0]),
                     "aktion_low": model.action_space.low.tolist(), "aktion_high": model.action_space.high.tolist()}}
    path = folder / snapshot_name(attempt)
    tmp = path.with_suffix(".tmp")
    torch.save(data, tmp)
    tmp.replace(path)
    return path


def load_snapshot(path):
    """-> (policy, meta). policy.predict(obs, deterministic=True) fährt wie das Modell zum Zeitpunkt des Snapshots."""
    import gymnasium as gym
    import torch
    from stable_baselines3.sac.policies import SACPolicy

    data = torch.load(path, map_location="cpu", weights_only=True)
    m = data["meta"]
    obs_space = gym.spaces.Box(sensors.LOW, sensors.HIGH, (m["obs_dim"],), np.float32)
    act_space = gym.spaces.Box(np.array(m["aktion_low"], np.float32), np.array(m["aktion_high"], np.float32))
    policy = SACPolicy(obs_space, act_space, lambda _: 0.0, net_arch=m["netz"])
    policy.actor.load_state_dict(data["actor"])
    policy.critic.load_state_dict(data["critic"])
    policy.set_training_mode(False)
    return policy, m


def snapshot_due(attempt, every, milestones):
    """Snapshot nach diesem Versuch? Alle `every` Versuche und zusätzlich bei den Meilensteinen."""
    return (every > 0 and attempt % every == 0) or attempt in milestones
