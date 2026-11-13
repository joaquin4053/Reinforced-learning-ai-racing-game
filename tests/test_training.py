"""Tests für das Training (Plan-Schritt 4).   Ausführen:  .venv/bin/python -m pytest tests/test_training.py"""
import csv
import json

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.buffers import ReplayBuffer

from rl.env import load_reward_config
from rl.recording import load_episode, load_snapshot, read_episode_meta, save_snapshot
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
         "ablauf": {"auswertung_alle_schritte": 600, "tensorboard_alle_schritte": 300},
         "aufzeichnung": {"episoden": True, "snapshot_alle_versuche": 2, "meilensteine": [1, 5]},
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

    rows = list(csv.DictReader(open(run.events_file)))
    assert [r["ereignis"] for r in rows if r["ereignis"] != "profilwechsel"] == ["pause", "pause"]
    evals = list(csv.DictReader(open(run.evals_file)))
    assert len(evals) == 4
    assert all(r["rundenzeit"] == "" or r["gueltig"] == "ja" for r in evals)   # Zeit nur bei gültiger Runde
    _check_recording(run, st2)
    live = json.loads(run.live_file.read_text())                       # für watch.py --live
    assert live["schritte"] == st2["letzte_auswertung"] and run.live_model.exists()
    json.loads((run.checkpoint_dir / "state.json").read_text())


def _check_recording(run, state):
    """Plan-Schritt 5: jeder Versuch genau einmal in summary.csv und episodes/, Snapshots nach Versuchen."""
    rows = list(csv.DictReader(open(run.summary_file)))
    nums = [int(r["versuch"]) for r in rows]
    assert nums == list(range(1, state["versuche"] + 1))                # fortlaufend, keine doppelt
    assert sum(r["art"] == "auswertung" for r in rows) == state["auswertungen"]
    files = sorted(run.episode_dir.glob("versuch_*.npz"))
    assert [int(f.stem.split("_")[1]) for f in files] == nums
    meta, arr = load_episode(files[-1])
    n = meta["fahrt_schritte"]
    assert arr["pos"].shape == (n + 1, 3) and arr["obs"].shape == (n + 1, 24)
    assert arr["aktion"].shape == (n, 4) and arr["reward"].shape == (n,)
    assert meta["reward"] == pytest.approx(float(arr["reward"].sum()), abs=0.5)
    assert read_episode_meta(files[-1]) == meta
    snaps = sorted(int(f.stem.split("_")[1]) for f in run.models_dir.glob("versuch_*.pt"))
    expected = [k for k in range(1, state["versuche"] + 1) if k % 2 == 0 or k in (1, 5)]
    assert snaps == expected
    assert rows[-1]["snapshot"] == f"versuch_{snaps[-1]:07d}.pt"


def test_snapshot_drives_like_model(tmp_path):
    """Ein Snapshot (nur Actor/Critic) fährt deterministisch genau wie das Modell."""
    from stable_baselines3 import SAC
    from rl.env import RaceEnv
    env = RaceEnv()
    model = SAC("MlpPolicy", env, policy_kwargs={"net_arch": [16, 16]}, seed=1, device="cpu")
    path = save_snapshot(tmp_path, 7, model, {"versuch": 7})
    policy, meta = load_snapshot(path)
    assert meta["versuch"] == 7 and path.stat().st_size < 100_000
    obs, _ = env.reset()
    for _ in range(50):
        a, _ = model.predict(obs, deterministic=True)
        b, _ = policy.predict(obs, deterministic=True)
        assert np.allclose(a, b, atol=1e-6)
        obs, *_ = env.step(a)


def test_resume_after_crash_never_reuses_attempt_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr("rl.run.RUNS_DIR", tmp_path)
    run = Run("t")
    run.create(SMALL)
    reward = load_reward_config()
    t1 = Trainer(run, SMALL, reward, RES, _profiles(1), "sparsam", stop_steps=300)
    t1.new(seed=0)
    t1.train({})
    run.log_attempt(versuch=t1.state["versuche"] + 10, art="training", schritte=999)   # lief nach der Pause weiter
    t2 = Trainer(run, SMALL, reward, RES, _profiles(1), "sparsam")
    t2.resume(run.find_checkpoint())
    assert t2.state["versuche"] == t1.state["versuche"] + 10


def test_migrate_old_summary(tmp_path, monkeypatch):
    monkeypatch.setattr("rl.run.RUNS_DIR", tmp_path)
    run = Run("t")
    run.create(SMALL)
    from rl.run import OLD_SUMMARY_FIELDS
    with open(run.summary_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OLD_SUMMARY_FIELDS)
        w.writeheader()
        w.writerow({"schritte": 0, "versuche": 0, "ereignis": "start"})
        w.writerow({"schritte": 25000, "versuche": 12, "ereignis": "auswertung", "gueltig": "ja", "rundenzeit": "55.10"})
    assert run.migrate_old_summary()
    assert not run.summary_file.exists() and (run.dir / "summary_alt.csv").exists()
    ev = list(csv.DictReader(open(run.evals_file)))
    assert ev[0]["versuch"] == "12" and ev[0]["rundenzeit"] == "55.10"
    assert [r["ereignis"] for r in csv.DictReader(open(run.events_file))] == ["start"]
    assert not run.migrate_old_summary()


def test_resume_keeps_records_from_after_checkpoint(tmp_path, monkeypatch):
    """Training ohne Pause beendet: Bestzeit/Stufen nach der letzten Sicherung dürfen nicht verloren gehen."""
    monkeypatch.setattr("rl.run.RUNS_DIR", tmp_path)
    run = Run("t")
    run.create(SMALL)
    reward = load_reward_config()
    t1 = Trainer(run, SMALL, reward, RES, _profiles(1), "sparsam", stop_steps=300)
    t1.new(seed=0)
    t1.train({})
    run.save_records({"bestzeit": 30.0, "bestzeit_schritte": 9999, "bestzeit_versuch": 50, "bestzeit_geist": "x.json",
                      "bestzeit_verlauf": [[9999, 30.0]], "stufen": {"4": {"schritte": 9999}}})
    t2 = Trainer(run, SMALL, reward, RES, _profiles(1), "sparsam")
    t2.resume(run.find_checkpoint())
    assert t2.state["bestzeit"] == 30.0 and "4" in t2.state["stufen"] and t2.stage == 4


def test_train_config_is_valid():
    """train_config.toml lässt sich lesen und hat alles, was ein neues Training braucht."""
    import tomllib
    from train import TRAIN_CONFIG
    with open(TRAIN_CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    for key in ("sac", "ablauf", "aufzeichnung", "erfolg", "ressourcen", "profile"):
        assert key in cfg
    assert {"episoden", "snapshot_alle_versuche", "meilensteine"} <= set(cfg["aufzeichnung"])


def test_start_from_existing_net(tmp_path, monkeypatch):
    """--start-netz: neues Training (Zähler bei 0, Buffer leer), aber mit den Gewichten des alten Modells."""
    import torch
    from rl.trainer import load_start_net
    monkeypatch.setattr("rl.run.RUNS_DIR", tmp_path)
    reward = load_reward_config()
    old = Run("alt")
    old.create(SMALL)
    t1 = Trainer(old, SMALL, reward, RES, _profiles(1), "sparsam", stop_steps=300)
    t1.new(seed=0)
    t1.train({})
    t1.model.save(tmp_path / "alt.zip")
    new = Run("neu")
    new.create(SMALL)
    t2 = Trainer(new, SMALL, reward, RES, _profiles(1), "sparsam")
    t2.new(seed=5, start_net=tmp_path / "alt.zip")
    assert t2.steps == 0 and t2.model.replay_buffer.size() == 0
    for a, b in zip(t1.model.policy.parameters(), t2.model.policy.parameters()):
        assert torch.equal(a, b)
    assert torch.equal(t1.model.log_ent_coef, t2.model.log_ent_coef)
    with pytest.raises(ValueError):                 # andere Aktionen (ohne Drift) -> passt nicht (R8)
        from stable_baselines3 import SAC
        from rl.env import RaceEnv
        cfg = {**reward, "umgebung": {**reward["umgebung"], "drift_erlaubt": False}}
        load_start_net(SAC("MlpPolicy", RaceEnv(config=cfg), device="cpu"), tmp_path / "alt.zip")


def test_finish_bonus_depends_on_lap_time():
    """Reward v2 (am 2026-10-19 ausprobiert, wieder zurückgenommen): Ziel-Bonus = 20 pro Sekunde unter dem
    Zeitlimit. Läufe mit v2 (v2_lauf3 bis 5) müssen sich weiter richtig ansehen lassen; v1 bleibt fest 100."""
    from rl.env import RaceEnv
    v2 = {**load_reward_config(), "gewichte": {"k_fortschritt": 1.0, "k_zeit": 0.1, "k_crash": 50.0,
                                                "k_ziel_pro_sekunde": 20.0}}
    env = RaceEnv(config=v2)
    assert env.finish_bonus(51.0) == pytest.approx(20 * (180 - 51))
    assert env.finish_bonus(48.0) - env.finish_bonus(55.0) == pytest.approx(140)
    assert env.finish_bonus(200.0) == 0.0
    assert RaceEnv().finish_bonus(51.0) == 100.0               # aktuell wieder Reward v1: fest 100


def test_start_net_warms_up_critic_first(tmp_path, monkeypatch):
    """--start-netz: Während des Aufwärmens ändert sich nur der Critic, danach lernt auch der Actor.
    Auch über Pausieren/Fortsetzen hinweg."""
    import copy
    import torch
    monkeypatch.setattr("rl.run.RUNS_DIR", tmp_path)
    reward = load_reward_config()
    cfg = copy.deepcopy(SMALL)
    cfg["sac"]["kritik_aufwaermen"] = 900
    old = Run("alt")
    old.create(cfg)
    t0 = Trainer(old, cfg, reward, RES, _profiles(1), "sparsam", stop_steps=300)
    t0.new(seed=0)
    t0.train({})
    t0.model.save(tmp_path / "alt.zip")
    actor0 = [p.detach().clone() for p in t0.model.actor.parameters()]
    critic0 = [p.detach().clone() for p in t0.model.critic.parameters()]
    run = Run("neu")
    run.create(cfg)
    t1 = Trainer(run, cfg, reward, RES, _profiles(1), "sparsam", stop_steps=600)
    t1.new(seed=1, start_net=tmp_path / "alt.zip")
    t1.train({})                                                   # 600 < 900: noch am Aufwärmen
    assert all(torch.equal(a, b) for a, b in zip(actor0, t1.model.actor.parameters()))
    assert not all(torch.equal(a, b) for a, b in zip(critic0, t1.model.critic.parameters()))
    t2 = Trainer(run, cfg, reward, RES, _profiles(1), "sparsam", stop_steps=1500)
    t2.resume(run.find_checkpoint())
    assert t2.frozen
    t2.train({})                                                   # über 900 hinaus: Actor lernt mit
    assert not t2.frozen
    assert not all(torch.equal(a, b) for a, b in zip(actor0, t2.model.actor.parameters()))
    assert "aufwaermen" in run.events_file.read_text()


def test_start_net_takes_weights_but_fresh_optimizers(tmp_path, monkeypatch):
    """Nur die Gewichte werden übernommen; Adam startet leer (sonst falscher Massstab bei neuem Reward)."""
    monkeypatch.setattr("rl.run.RUNS_DIR", tmp_path)
    reward = load_reward_config()
    old = Run("alt")
    old.create(SMALL)
    t0 = Trainer(old, SMALL, reward, RES, _profiles(1), "sparsam", stop_steps=400)
    t0.new(seed=0)
    t0.train({})
    assert t0.model.actor.optimizer.state                       # das alte Modell hat gelernt
    t0.model.save(tmp_path / "alt.zip")
    run = Run("neu")
    run.create(SMALL)
    t1 = Trainer(run, SMALL, reward, RES, _profiles(1), "sparsam")
    t1.new(seed=1, start_net=tmp_path / "alt.zip")
    for opt in (t1.model.actor.optimizer, t1.model.critic.optimizer, t1.model.ent_coef_optimizer):
        assert not opt.state                                   # frisch


def test_action_repeat_keeps_physics_time_and_reward():
    """Aktion 3 Physik-Schritte halten: gleiche Fahrt, gleiche Zeit (R6), Reward = Summe (R7), jedes Bild aufgezeichnet."""
    from rl.env import RaceEnv
    a = RaceEnv(aufzeichnen=True)
    b = RaceEnv(aufzeichnen=True, wiederholen=3)
    frames = []
    b.on_frame = lambda act, r, obs: frames.append(r)
    a.reset(); b.reset()
    act = np.array([0.2, 1.0, 0.0, 0.0])
    ra = sum(a.step(act)[1] for _ in range(30))
    rb = sum(b.step(act)[1] for _ in range(10))
    assert a.sim.frame == b.sim.frame == 30
    assert (a.sim.car.x, a.sim.car.z) == pytest.approx((b.sim.car.x, b.sim.car.z))
    assert ra == pytest.approx(rb) and len(frames) == 30 and sum(frames) == pytest.approx(rb)
    assert len(b.recorder.actions) == 30


def test_variant_feinschliff_in_config():
    import tomllib
    from train import TRAIN_CONFIG
    v = tomllib.load(open(TRAIN_CONFIG, "rb"))["varianten"]["feinschliff"]
    assert v["aktion_wiederholen"] == 3 and v["gamma"] == pytest.approx(0.995 ** 3, abs=1e-3)
