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
