"""Tests für die Dashboard-Zahlen (Plan-Schritt 8).   .venv/bin/python -m pytest tests/test_stats.py"""
import numpy as np
import pytest

from rl.playback import Attempt
from rl.stats import moving_average, official_best, reward_parts, share_valid

W = {"k_fortschritt": 1.0, "k_zeit": 0.1, "k_crash": 50.0, "k_ziel": 100.0}


def _a(n, result="Crash", kind="training", t=None, prog=40.0):
    return Attempt(n, kind, n * 100, result, result, 0.0, t, 100.0 if result == "Ziel" else prog, 1.0)


def test_moving_average():
    assert moving_average([]).size == 0
    np.testing.assert_allclose(moving_average([2, 4, 6, 8], window=2), [2, 3, 5, 7])


@pytest.mark.parametrize("result,last", [("Crash", -50.0), ("Ziel", 100.0), ("Abbruch", 0.0)])
def test_reward_parts_add_up_and_follow_r7(result, last):
    progress = np.array([0.0, 1.5, 2.0, 0.0])                 # m neuer Höchststand je Schritt
    r = progress - 0.1
    r[-1] += last
    parts = reward_parts(r, W, result)
    np.testing.assert_allclose(parts["fortschritt"], progress, atol=1e-9)
    np.testing.assert_allclose(parts["zeit"], -0.1)
    total = parts["fortschritt"] + parts["zeit"] + parts["crash"] + parts["ziel"]
    np.testing.assert_allclose(total, r)
    assert parts["crash"][-1] == (-50.0 if result == "Crash" else 0.0)
    assert parts["ziel"][-1] == (100.0 if result == "Ziel" else 0.0)


def test_official_best_only_from_evaluations():
    attempts = [_a(1, "Ziel", t=50.0),                         # Training: zählt nicht (R6)
                _a(2, "Ziel", "auswertung", 60.0), _a(3, "Crash", "auswertung"),
                _a(4, "Ziel", "auswertung", 58.0), _a(5, "Ziel", "auswertung", 59.0)]
    assert official_best(attempts) == [(2, 60.0), (4, 58.0)]


def test_share_valid():
    attempts = [_a(i, "Ziel" if i % 4 == 0 else "Crash", t=50.0) for i in range(1, 201)]
    assert share_valid(attempts) == (25.0, 100)
    assert share_valid([]) == (0.0, 0)


def test_numpy_actor_matches_model_predict(tmp_path):
    """Das Netz im Dashboard rechnet genau wie model.predict(deterministic=True) – aus Modell und aus Snapshot."""
    from stable_baselines3 import SAC

    from rl.env import RaceEnv
    from rl.netview import Actor, nearest_snapshot
    from rl.recording import save_snapshot
    env = RaceEnv()
    model = SAC("MlpPolicy", env, policy_kwargs={"net_arch": [64, 64]}, seed=2, device="cpu")
    save_snapshot(tmp_path, 40, model, {"versuch": 40})
    save_snapshot(tmp_path, 60, model, {"versuch": 60})
    assert nearest_snapshot(tmp_path, 55).name == "versuch_0000040.pt"
    assert nearest_snapshot(tmp_path, 10).name == "versuch_0000040.pt"
    for actor in (Actor.from_policy(model.policy), Actor.from_snapshot(tmp_path / "versuch_0000060.pt")):
        obs, _ = env.reset()
        for _ in range(30):
            want, _ = model.predict(obs, deterministic=True)
            h1, h2, got = actor.forward(obs)
            assert h1.shape == (64,) and h2.shape == (64,) and (h1 >= 0).all()
            np.testing.assert_allclose(got, want, atol=1e-5)
            obs, *_ = env.step(want)


def test_grouping():
    from rl.netview import group, group_weights
    np.testing.assert_allclose(group(np.arange(8), 4), [0.5, 2.5, 4.5, 6.5])
    w = np.ones((64, 24))
    g = group_weights(w, 16, 6)
    assert g.shape == (16, 6) and g[0, 0] == 4 * 4


def test_numpy_critic_and_contributions_match_model(tmp_path):
    """Critic im Dashboard = min(Q1, Q2) von SB3 für die Actor-Aktion; Beiträge = Gewicht x Aktivierung."""
    import torch
    from stable_baselines3 import SAC

    from rl.env import RaceEnv
    from rl.netview import INPUT_NAMES, load_snapshot_nets, snapshot_before
    from rl.recording import save_snapshot
    env = RaceEnv()
    model = SAC("MlpPolicy", env, policy_kwargs={"net_arch": [64, 64]}, seed=4, device="cpu")
    save_snapshot(tmp_path, 20, model, {"versuch": 20})
    path = save_snapshot(tmp_path, 40, model, {"versuch": 40})
    assert snapshot_before(tmp_path, path).name == "versuch_0000020.pt"
    assert snapshot_before(tmp_path, tmp_path / "versuch_0000020.pt") is None
    actor, critic = load_snapshot_nets(path)
    obs = np.stack([env.reset()[0]] + [env.step(env.action_space.sample())[0] for _ in range(20)])
    h1, h2, squashed, act = actor.forward_batch(obs)
    with torch.no_grad():
        q1, q2 = model.critic(torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(squashed, dtype=torch.float32))
    np.testing.assert_allclose(critic.value(obs, squashed), torch.min(q1, q2)[:, 0].numpy(), atol=1e-4)
    c1, c2, c3 = actor.contributions(obs[3], h1[3], h2[3])
    assert c1.shape == (64, 24) and c2.shape == (64, 64) and c3.shape == (4, 64)
    np.testing.assert_allclose(c1.sum(axis=1) + actor.b1, actor.w1 @ obs[3] + actor.b1)   # Summe der Beiträge = Eingang
    assert len(INPUT_NAMES) == 24 and INPUT_NAMES[8] == "Sensor vorne" and INPUT_NAMES[0] == "Sensor 90° L"


def test_reward_parts_v2_time_bonus():
    w2 = {"k_fortschritt": 1.0, "k_zeit": 0.1, "k_crash": 50.0, "k_ziel_pro_sekunde": 20.0, "zeitlimit": 180.0}
    n = 60 * 50                                         # Runde in 50 s
    r = np.full(n, 0.9)
    r[-1] += 20 * (180 - 50)
    parts = reward_parts(r, w2, "Ziel")
    assert parts["ziel"][-1] == pytest.approx(2600)
    np.testing.assert_allclose(parts["fortschritt"], 1.0, atol=1e-9)
