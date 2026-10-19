"""Die Netze der KI nachrechnen, um sie im Dashboard zu zeigen (Plan-Schritte 8 und 9) – mit numpy, ohne Training.

Actor (entscheidet), SAC aus Stable-Baselines3, siehe train_config.toml [sac] netz:
    24 Beobachtungen -> 64 Neuronen (ReLU) -> 64 Neuronen (ReLU) -> 4 Werte -> tanh -> Aktion
    Aktion = untere Grenze + (tanh + 1) / 2 * (obere - untere)    (Lenkung -1..1, Gas/Bremse/Drift 0..1)
    Das ist genau, was model.predict(obs, deterministic=True) rechnet (Test in tests/test_stats.py).
Critic (bewertet): zwei Netze Q1, Q2 mit 24 Beobachtungen + 4 Aktionen (tanh-Werte) -> 64 -> 64 -> 1.
    "Wert der Situation" = min(Q1, Q2) für die Aktion, die der Actor wählen würde – so rechnet SAC selbst.
Die gespeicherten Beobachtungen eines Versuchs werden durch den passenden Snapshot geschickt; trainiert wird nichts.
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
        h1, h2, _, act = self.forward_batch(np.asarray(obs, dtype=np.float64)[None, :])
        return h1[0], h2[0], act[0]

    def forward_batch(self, obs):
        """Viele Beobachtungen auf einmal (n x 24) -> (Schicht 1, Schicht 2, tanh-Werte, Aktion), je n Zeilen."""
        x = np.asarray(obs, dtype=np.float64)
        h1 = np.maximum(0.0, x @ self.w1.T + self.b1)
        h2 = np.maximum(0.0, h1 @ self.w2.T + self.b2)
        squashed = np.tanh(h2 @ self.w3.T + self.b3)
        return h1, h2, squashed, self.low + (squashed + 1.0) / 2.0 * (self.high - self.low)

    def contributions(self, obs, h1, h2):
        """Beitrag jeder Verbindung = Gewicht x Aktivierung des Neurons, von dem sie kommt.
        -> (Eingaben->Schicht 1: 64x24, Schicht 1->2: 64x64, Schicht 2->Ausgaben: 4x64)"""
        return self.w1 * np.asarray(obs, dtype=np.float64)[None, :], self.w2 * h1[None, :], self.w3 * h2[None, :]

    @staticmethod
    def from_policy(policy, label=""):
        """Aus einem geladenen Modell (watch.py: das Netz, das gerade fährt)."""
        sd = {k: v.detach().cpu().numpy() for k, v in policy.actor.state_dict().items()}
        space = policy.action_space
        return Actor(sd, space.low, space.high, label)

    @staticmethod
    def from_snapshot(path):
        """Aus einem Snapshot in runs/<name>/models/ (torch wird erst hier geladen)."""
        return load_snapshot_nets(path)[0]


class Critic:
    """Bewertet eine Situation: Wie viel Reward erwartet die KI ab hier noch (abgezinst mit gamma)?"""

    def __init__(self, weights):
        w = {k: np.asarray(v, dtype=np.float64) for k, v in weights.items()}
        self.nets = [[(w[f"qf{q}.{i}.weight"], w[f"qf{q}.{i}.bias"]) for i in (0, 2, 4)] for q in (0, 1)]

    def value(self, obs, squashed):
        """(n x 24 Beobachtungen, n x 4 tanh-Aktionen) -> n Werte = min(Q1, Q2)."""
        x = np.concatenate([np.asarray(obs, np.float64), np.asarray(squashed, np.float64)], axis=1)
        qs = []
        for (w0, b0), (w1, b1), (w2, b2) in self.nets:
            h = np.maximum(0.0, x @ w0.T + b0)
            h = np.maximum(0.0, h @ w1.T + b1)
            qs.append((h @ w2.T + b2)[:, 0])
        return np.minimum(*qs)

    @staticmethod
    def from_policy(policy):
        return Critic({k: v.detach().cpu().numpy() for k, v in policy.critic.state_dict().items()})


def load_snapshot_nets(path):
    """-> (Actor, Critic) aus einem Snapshot (ein einziges Laden)."""
    import torch
    data = torch.load(path, map_location="cpu", weights_only=True)
    m = data["meta"]
    actor = Actor({k: v.numpy() for k, v in data["actor"].items()}, m["aktion_low"], m["aktion_high"],
                  f"Snapshot nach Versuch {m['versuch']:,}".replace(",", "'"))
    actor.versuch = m["versuch"]
    critic = Critic({k: v.numpy() for k, v in data["critic"].items()}) if data.get("critic") else None
    return actor, critic


def snapshot_files(models_dir):
    """[(Versuch, Pfad)] aller Snapshots, aufsteigend."""
    out = []
    for f in Path(models_dir).glob("versuch_*.pt"):
        try:
            out.append((int(f.stem.split("_")[1]), f))
        except (IndexError, ValueError):
            pass
    return sorted(out)


def snapshot_before(models_dir, path):
    """Der Snapshot direkt vor diesem (für "Veränderung zeigen") oder None."""
    files = snapshot_files(models_dir)
    i = next((k for k, (_, f) in enumerate(files) if Path(f).name == Path(path).name), None)
    return files[i - 1][1] if i else None


SENSOR_NAMES = [f"Sensor {abs(a):.0f}° {'L' if a < 0 else 'R'}" if abs(a) > 0.5 else "Sensor vorne"
                for a in np.degrees(np.linspace(-np.pi / 2, np.pi / 2, 17))]
INPUT_NAMES = SENSOR_NAMES + ["Tempo", "Winkel", "Rutschen", "Lenkrad", "Steigung", "In der Luft", "Senkrecht"]
OUTPUT_NAMES = ["Lenkung", "Gas", "Bremse", "Drift"]


def nearest_snapshot(models_dir, versuch):
    """Snapshot, der dem Versuch am nächsten liegt (lieber davor: so ist das Netz gefahren). -> Pfad oder None."""
    files = snapshot_files(models_dir)
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
