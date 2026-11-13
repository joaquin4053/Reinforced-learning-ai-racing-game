"""Alle Strecken laden sich, ohne Überlappung, mit Checkpoints in Reihenfolge; Kulisse ist gültig."""
from pathlib import Path

import numpy as np
import pytest

from game.track import Track
from render.deko import ARTEN

TRACKS = sorted(Path("tracks").glob("*.json"))


@pytest.mark.parametrize("path", TRACKS, ids=lambda p: p.stem)
def test_track_loads(path):
    t = Track.load(path)
    assert t.length > 500 and t.checkpoints
    assert all(d.get("art") in ARTEN for d in t.deko)
    assert np.isfinite(t.start_y)


def test_sommer5():
    """Nachbau von PolyTrack "Summer 5" (beobachtet, eigene Umsetzung): 8 Checkpoints, ~3.9 km."""
    t = Track.load("tracks/poly04.json")
    assert len(t.checkpoints) == 8
    assert 3800 < t.length < 4050
    assert np.degrees(np.arctan(np.abs(t.samples["tilt"]))).max() < 45     # steiler wäre eine Wand (Physik)
