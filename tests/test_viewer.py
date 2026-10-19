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


# ---------- Geister (Plan-Schritt 7) ----------
def _many(run, n, frames=lambda i: 30 + i):
    for i in range(1, n + 1):
        _episode(run, i, frames(i), "Ziel" if i % 3 == 0 else "Crash")


def test_range_spreads_ghosts_evenly(run):
    _many(run, 400)
    p = Player(EpisodeIndex(run.dir))
    p.set_ghosts(n=30)
    p.set_range(100, 300)                                    # 30 Geister aus Versuch 100–300
    nums = [a.versuch for a, _ in p.group]
    assert len(nums) == 30 and nums[0] == 100 and nums[-1] == 300 and nums == sorted(nums)
    gaps = np.diff(nums)
    assert gaps.max() - gaps.min() <= 1                      # gleichmässig verteilt
    best = min((a for a, _ in p.group), key=Player.best_key)
    assert p.current.versuch == best.versuch and best.gueltig and not p.live   # Kamera beim Besten
    p.set_range(1000, 500)                                   # Reihenfolge egal
    assert p.range == (500, 1000) and [a.versuch for a, _ in p.group][-1] == 400   # nur, was es gibt
    p.set_range(600, 900)                                    # kein Versuch im Bereich: der neueste bleibt
    assert [a.versuch for a, _ in p.group] == [400]


def test_range_smaller_than_count_takes_all(run):
    _many(run, 60)
    p = Player(EpisodeIndex(run.dir))
    p.set_ghosts(n=30)
    p.set_range(10, 19)
    assert [a.versuch for a, _ in p.group] == list(range(10, 20))
    p.focus_on(12)                                           # Klick auf einen Geist
    assert p.current.versuch == 12 and len(p.group) == 10
    p.focus_on(25)                                           # nicht in der Gruppe -> nichts
    assert p.current.versuch == 12
    p.step_attempt(1)                                        # N: nächster Bereich gleicher Breite
    assert p.range == (20, 29) and [a.versuch for a, _ in p.group] == list(range(20, 30))


def test_click_on_timeline_moves_range(run):
    _many(run, 100, frames=lambda i: 30)
    p = Player(EpisodeIndex(run.dir))
    p.set_ghosts(n=5)
    p.set_range(10, 30)
    p.select_versuch(50)                                     # Klick auf die Zeitleiste
    assert p.range == (50, 70) and 50 <= p.current.versuch <= 70
    for _ in range(40):                                      # Gruppe fertig -> anhalten (nicht weiterspringen)
        p.update(0.05)
    assert not p.playing and p.range == (50, 70)


def test_live_with_last_n(run):
    _many(run, 20, frames=lambda i: 30)
    p = Player(EpisodeIndex(run.dir))
    p.set_ghosts(n=5)
    assert p.live and [a.versuch for a, _ in p.group] == [16, 17, 18, 19, 20] and p.current.gueltig
    for i in range(21, 24):
        _episode(run, i, 30)
    p.refresh()
    for _ in range(60):
        p.update(0.05)
    assert [a.versuch for a, _ in p.group] == [19, 20, 21, 22, 23] and p.live
    p.select_versuch(5)                                      # Klick in die Vergangenheit
    assert not p.live and p.group_mode == "bereich" and [a.versuch for a, _ in p.group][0] == 5


def test_milestones(run):
    _many(run, 120, frames=lambda i: 30)
    p = Player(EpisodeIndex(run.dir))
    p.set_ghosts(mode="meilensteine")
    assert [a.versuch for a, _ in p.group] == [10, 20, 50, 100]
    p.select_versuch(20)                                     # Meilenstein angeklickt: nur Fokus
    assert p.current.versuch == 20 and len(p.group) == 4
    p.select_versuch(33)                                     # anderer Versuch: Bereich ab dort
    assert p.group_mode == "bereich" and p.current.versuch == 33


def test_ghosts_off_is_single_attempt(run):
    _many(run, 10, frames=lambda i: 30)
    p = Player(EpisodeIndex(run.dir))
    p.set_ghosts(n=5)
    p.set_ghosts(n=1)
    assert len(p.group) == 1 and p.group[0][0].versuch == p.current.versuch == 10


def test_camera_follows_the_one_who_gets_furthest(run):
    """Mit Geistern folgt die Kamera dem, der am weitesten kommt: im Ziel der schnellste, sonst der mit dem
    meisten Fortschritt – egal welcher Versuch es ist. Ein Klick auf einen anderen gilt bis zur nächsten Gruppe."""
    def add(n, result, frames, pct):
        _episode(run, n, frames, result)
        rows = run.summary_file.read_text().splitlines()
        rows[-1] = rows[-1].replace(",40.00,", f",{pct:.2f},") if result == "Crash" else rows[-1]
        run.summary_file.write_text("\n".join(rows) + "\n")
    add(1, "Crash", 30, 40.0)
    add(2, "Crash", 30, 85.0)          # kommt am weitesten, aber nicht ins Ziel
    add(3, "Crash", 30, 60.0)
    p = Player(EpisodeIndex(run.dir))
    p.set_ghosts(n=5)
    p.set_range(1, 3)
    assert p.current.versuch == 2
    add(4, "Ziel", 90, 100.0)          # im Ziel schlägt jeden Crash
    add(5, "Ziel", 60, 100.0)          # ... und der schnellere gewinnt
    p.refresh()
    p.set_range(1, 5)
    assert p.current.versuch == 5
    p.focus_on(1)                      # Klick auf einen anderen Geist
    assert p.current.versuch == 1
    p.set_range(1, 4)                  # neue Gruppe: wieder der Beste
    assert p.current.versuch == 4
