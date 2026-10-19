"""Das Netz der KI (Actor) nachrechnen, um es im Dashboard zu zeigen (Plan-Schritt 8) – mit numpy, ohne Training.

Aufbau (SAC aus Stable-Baselines3, siehe train_config.toml [sac] netz):
    24 Beobachtungen -> 64 Neuronen (ReLU) -> 64 Neuronen (ReLU) -> 4 Werte -> tanh -> Aktion
    Aktion = untere Grenze + (tanh + 1) / 2 * (obere - untere)    (Lenkung -1..1, Gas/Bremse/Drift 0..1)
Das ist genau, was model.predict(obs, deterministic=True) rechnet (Test in tests/test_stats.py).
"""
from pathlib import Path

import numpy as np


class Actor:
    def __init__(self, weights, low, high, label=""):
        w = {k: np.asarray(v, dtype=np.float64) for k, v in weights.items()}
        self.w1, self.b1 = w["latent_pi.0.weight"], w["latent_pi.0.bias"]
        self.w2, self.b2 = w["latent_pi.2.weight"], w["latent_pi.2.bias"]
        self.w3, self.b3 = w["mu.weight"], w["mu.bias"]
        self.low, self.high = np.asarray(low, np.float64), np.asarray(high, np.float64)
        self.label = label

    def forward(self, obs):
        """-> (Schicht 1, Schicht 2, Aktion): die Aktivierungen jedes Neurons und die Aktion."""
        x = np.asarray(obs, dtype=np.float64)
        h1 = np.maximum(0.0, self.w1 @ x + self.b1)
        h2 = np.maximum(0.0, self.w2 @ h1 + self.b2)
        a = np.tanh(self.w3 @ h2 + self.b3)
        return h1, h2, self.low + (a + 1.0) / 2.0 * (self.high - self.low)

    @staticmethod
    def from_policy(policy, label=""):
        """Aus einem geladenen Modell (watch.py: das Netz, das gerade fährt)."""
        sd = {k: v.detach().cpu().numpy() for k, v in policy.actor.state_dict().items()}
        space = policy.action_space
        return Actor(sd, space.low, space.high, label)

    @staticmethod
    def from_snapshot(path):
        """Aus einem Snapshot in runs/<name>/models/ (torch wird erst hier geladen)."""
        import torch
        data = torch.load(path, map_location="cpu", weights_only=True)
        m = data["meta"]
        sd = {k: v.numpy() for k, v in data["actor"].items()}
        return Actor(sd, m["aktion_low"], m["aktion_high"], f"Snapshot nach Versuch {m['versuch']:,}".replace(",", "'"))


def nearest_snapshot(models_dir, versuch):
    """Snapshot, der dem Versuch am nächsten liegt (lieber davor: so ist das Netz gefahren). -> Pfad oder None."""
    files = []
    for f in Path(models_dir).glob("versuch_*.pt"):
        try:
            files.append((int(f.stem.split("_")[1]), f))
        except (IndexError, ValueError):
            pass
    if not files:
        return None
    before = [x for x in files if x[0] <= versuch]
    return (max(before) if before else min(files))[1]


def group(values, n):
    """Neuronen zu n Gruppen zusammenfassen (Mittelwert) – für eine einfache Darstellung."""
    v = np.asarray(values, dtype=np.float64)
    return np.array([c.mean() for c in np.array_split(v, n)])


def group_weights(w, n_out, n_in):
    """Gewichte (out x in) zu (n_out x n_in) Gruppen zusammenfassen (Summe)."""
    rows = np.array_split(np.arange(w.shape[0]), n_out)
    cols = np.array_split(np.arange(w.shape[1]), n_in)
    return np.array([[w[np.ix_(r, c)].sum() for c in cols] for r in rows])
