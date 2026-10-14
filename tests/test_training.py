"""Tests für das Training (Plan-Schritt 4).   Ausführen:  .venv/bin/python -m pytest tests/test_training.py"""
import csv
import json

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.buffers import ReplayBuffer

from rl.env import load_reward_config
from rl.run import Run
from rl.trainer import Trainer, regroup_buffer, update_schedule


@pytest.mark.parametrize("envs", [1, 2, 3, 4, 5, 9])
def test_update_schedule_keeps_ratio(envs):
    freq, grads = update_schedule(envs, 0.25)
    assert grads / (freq * envs) == pytest.approx(0.25)


def _filled_buffer(n_envs, size, steps):
    obs = gym.spaces.Box(-1, 1, (3,), np.float32)
    act = gym.spaces.Box(-1, 1, (2,), np.float32)
    buf = ReplayBuffer(size, obs, act, device="cpu", n_envs=n_envs)
    for t in range(steps):
        ids = np.array([t * 10 + e for e in range(n_envs)], np.float32)   # jeder Schritt eindeutig
        o = np.repeat(ids[:, None], 3, axis=1)
        buf.add(o, o + 0.5, np.zeros((n_envs, 2)), ids, np.zeros(n_envs), [{} for _ in range(n_envs)])
    return buf


def _ids(buf):
    rows = buf.buffer_size if buf.full else buf.pos
    return sorted(buf.rewards[:rows].ravel().tolist())


@pytest.mark.parametrize("new_envs", [1, 2, 4])
def test_regroup_keeps_all_steps(new_envs):
    old = _filled_buffer(3, 300, 40)                 # 120 Schritte, Platz für 300
    new = regroup_buffer(old, new_envs)
    assert new.n_envs == new_envs and new.buffer_size * new_envs <= 300
    kept = _ids(new)
    assert len(kept) == 120 - 120 % new_envs
    assert set(kept) <= set(_ids(old))
    rows = new.pos
    assert np.allclose(new.next_observations[:rows, :, 0], new.observations[:rows, :, 0] + 0.5)  # Paare bleiben


def test_regroup_full_buffer_keeps_newest():
    old = _filled_buffer(2, 100, 80)                  # 50 Zeilen Platz, 80 geschrieben -> voll, Ringpuffer
    new = regroup_buffer(old, 4)
    assert new.full
    assert _ids(new) == _ids(old)                     # alle 100 Schritte, nichts verloren
    assert min(_ids(new)) == 300                      # die ältesten (t < 30) wurden überschrieben


SMALL = {"sac": {"lernrate": 3e-4, "puffer": 3000, "batch": 32, "gamma": 0.99, "tau": 0.005, "lernstart": 200,
                 "updates_pro_schritt": 0.25, "netz": [16, 16]},
         "ablauf": {"auswertung_alle_schritte": 600, "snapshot_alle_schritte": 1000, "tensorboard_alle_schritte": 300},
         "erfolg": {"zuverlaessig_serie": 5, "plateau_schritte": 10**9, "plateau_sekunden": 0.1},
         "versionen": {}, "strecke_id": "map01"}
RES = {"sicherung_minuten": 30, "akku_minimum": 0, "speicher_minimum_gb": 0, "pruefen_sekunden": 3600}


def _profiles(envs):     # alle Profile gleich, damit der Test nicht davon abhängt, ob der Mac am Netz ist
    return {p: {"umgebungen": envs, "threads": 1, "prioritaet": 0} for p in ("sparsam", "normal", "voll")}


def test_pause_and_resume_with_other_profile(tmp_path, monkeypatch):
    monkeypatch.setattr("rl.run.RUNS_DIR", tmp_path)
    run = Run("t")
    run.create(SMALL)
    reward = load_reward_config()

    t1 = Trainer(run, SMALL, reward, RES, _profiles(3), "normal", stop_steps=1500)
    t1.new(seed=0)
    t1.train({"start": "1"})
    st1 = run.load_state(run.find_checkpoint())
    assert st1["schritte"] >= 1500 and st1["auswertungen"] == 2
    assert t1.model.replay_buffer.n_envs == 3

    t2 = Trainer(run, SMALL, reward, RES, _profiles(2), "normal", stop_steps=2400)
    t2.resume(run.find_checkpoint())
    assert t2.model.num_timesteps == st1["schritte"]                 # genau dort weiter
    stored = t2.model.replay_buffer
    assert stored.n_envs == 2 and stored.pos * 2 >= st1["schritte"] - 6  # Erfahrung behalten (bis auf den letzten Schritt vor der Pause)
    t2.train({"start": "2"})
    st2 = run.load_state(run.find_checkpoint())
    assert st2["schritte"] >= 2400 and st2["auswertungen"] == 4 and st2["versuche"] >= st1["versuche"]
    assert len(st2["sitzungen"]) == 2

    rows = list(csv.DictReader(open(run.summary_file)))
    assert [r["ereignis"] for r in rows if r["ereignis"] not in ("auswertung", "profilwechsel")] == ["pause", "pause"]
    assert all(r["rundenzeit"] == "" or r["gueltig"] == "ja" for r in rows)   # Zeit nur bei gültiger Runde
    assert list((run.dir / "snapshots").glob("*.zip"))
    json.loads((run.checkpoint_dir / "state.json").read_text())
