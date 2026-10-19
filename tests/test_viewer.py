"""Tests für die Wiedergabe im Viewer (Plan-Schritt 6), ohne Grafik.   .venv/bin/python -m pytest tests/test_viewer.py"""
import numpy as np
import pytest

from game.car import DT
from rl.playback import EpisodeIndex, Player
from rl.recording import save_episode
from rl.run import Run


def _episode(run, n, frames, result="Crash", kind="training"):
    """Versuch n wie vom Training: erst die Datei, dann die Zeile in summary.csv."""
    pos = np.zeros((frames + 1, 3), np.float32)
    pos[:, 2] = np.arange(frames + 1) * 0.5                       # fährt geradeaus, 30 m/s
    rot = np.zeros((frames + 1, 3), np.float16)
    obs = np.zeros((frames + 1, 24), np.float16)
    arrays = {"pos": pos, "rot": rot, "obs": obs, "aktion": np.zeros((frames, 4), np.float16),
              "reward": np.full(frames, 0.4, np.float16)}
    save_episode(run.episode_dir, n, arrays, {"versuch": n})
    valid = result == "Ziel"
    run.log_attempt(versuch=n, art=kind, schritte=n * 100, ergebnis=result, ende="Ziel" if valid else "Stillstand",
                    reward=frames * 0.4, rundenzeit=frames * DT if valid else None,
                    fortschritt_pct=100.0 if valid else 40.0, dauer_s=frames * DT, fahrt_schritte=frames)


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr("rl.run.RUNS_DIR", tmp_path)
    r = Run("t")
    r.create({"versionen": {}})
    return r


def test_index_reads_only_new_rows_and_skips_half_written(run):
    _episode(run, 1, 60)
    _episode(run, 2, 60, "Ziel")
    idx = EpisodeIndex(run.dir)
    assert idx.refresh() == 2 and [a.versuch for a in idx.attempts] == [1, 2]
    assert idx.attempts[1].gueltig and idx.attempts[1].rundenzeit == pytest.approx(1.0)
    with open(run.summary_file, "a") as f:            # Training schreibt gerade eine Zeile
        f.write("2026-10-19 10:00:00,3,training")
    assert idx.refresh() == 0
    with open(run.summary_file, "a") as f:
        f.write(",300,Crash,Stillstand,1.0,,10.00,1.00,60,0,\n")
    assert idx.refresh() == 1 and idx.attempts[-1].versuch == 3
    assert idx.load(3) is None                          # Datei fehlt -> kein Absturz
    assert idx.load(1).frames == 60


def test_index_of_run_without_recording(run):
    run.summary_file.write_text("zeit,laufzeit_s,schritte,versuche,ereignis\n2026,0,0,0,start\n")
    idx = EpisodeIndex(run.dir)
    idx.refresh()
    assert idx.attempts == []
    assert Player(idx).current is None


def test_pose_interpolates_between_steps(run):
    _episode(run, 1, 60)
    ep = EpisodeIndex(run.dir).load(1)
    assert ep.duration == pytest.approx(1.0)
    assert ep.pose(0.0)[2] == 0.0
    assert ep.pose(1.5 * DT)[2] == pytest.approx(0.75)           # zwischen Schritt 1 und 2
    assert ep.pose(99.0)[2] == pytest.approx(30.0)               # nach dem Ende: letzte Stelle
    assert ep.reward_until(0.5) == pytest.approx(30 * 0.4, rel=1e-3)


def test_live_follows_training(run):
    _episode(run, 1, 60)
    p = Player(EpisodeIndex(run.dir))
    assert p.live and p.current.versuch == 1
    _episode(run, 2, 60)
    _episode(run, 3, 60)
    p.refresh()
    assert p.current.versuch == 1                     # läuft erst fertig ...
    for _ in range(200):
        p.update(0.05)
    assert p.current.versuch == 3                     # ... dann direkt der neueste
    assert p.live


def test_not_live_plays_next_and_stops_at_the_end(run):
    for n in (1, 2, 3):
        _episode(run, n, 30)
    p = Player(EpisodeIndex(run.dir))
    p.select(0)
    assert not p.live
    seen = set()
    for _ in range(300):
        p.update(0.05)
        seen.add(p.current.versuch)
    assert seen == {1, 2, 3} and not p.playing


def test_speed_seek_and_filter(run):
    for n, res, kind in ((1, "Crash", "training"), (2, "Ziel", "training"), (3, "Ziel", "auswertung"),
                         (4, "Crash", "auswertung")):
        _episode(run, n, 120, res, kind)
    p = Player(EpisodeIndex(run.dir))
    p.select(0)
    p.faster(3)                                       # 1× -> 10×
    assert p.speed == 10
    p.update(0.1)
    assert p.t == pytest.approx(1.0)
    p.seek(-5)
    assert p.t == 0.0
    p.set_filter("gueltige")
    assert [a.versuch for a in p.list] == [2, 3] and p.current.versuch == 2
    p.set_filter("auswertungen")
    assert [a.versuch for a in p.list] == [3, 4] and p.current.versuch == 3
    p.go_live()
    assert p.current.versuch == 4 and p.live
    p.step_attempt(-1)
    assert p.current.versuch == 3 and not p.live


def test_live_episode_drives_exactly_like_the_evaluation():
    """watch.py: Eine Version fährt jetzt genau die Fahrt ihrer Auswertung (deterministisch, R6),
    auch wenn sie in kleinen Stücken (pro Bild) weiterfährt und dazwischen zurückgespult wird."""
    from stable_baselines3 import SAC

    from rl.env import RaceEnv
    from rl.playback import LiveEpisode
    from rl.trainer import evaluate
    model = SAC("MlpPolicy", RaceEnv(), policy_kwargs={"net_arch": [16, 16]}, seed=3, device="cpu")
    info, sim = evaluate(model, None, record=False)
    ep = LiveEpisode(model, RaceEnv(), expected=info["schritte"] * DT)
    t = 0.0
    while not ep.done:
        t += 0.37
        ep.extend(t)
        assert ep.duration <= t + DT
        ep.pose(t / 2)                                   # zurückspulen: nur lesen, nichts neu rechnen
    assert ep.frames == info["schritte"] and ep.info["ende"] == info["ende"]
    x, y, z = sim.car.x, sim.car.y, sim.car.z
    assert np.allclose(ep.pos[ep.frames], (x, y, z), atol=1e-3)
