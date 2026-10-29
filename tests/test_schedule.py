"""The diffusion-time schedule, with and without the mid-anneal plateau."""

import numpy as np
import pytest

from novamac.sampler import build_schedule


def test_schedule_without_plateau():
    ts, p0 = build_schedule(0.6, 4, 10, 3)
    assert p0 == -1 and len(ts) == 17
    assert np.all(ts[:4] == 0.6) and np.all(ts[-3:] == 0) and np.all(np.diff(ts[4:14]) < 0)
    assert ts[13] == 0


def test_schedule_with_plateau():
    ts, p0 = build_schedule(0.6, 4, 10, 3, plateau_at=0.5, plateau_steps=5)
    assert len(ts) == 4 + 10 + 5 + 3
    plateau = ts[p0 : p0 + 5]
    assert np.all(plateau == plateau[0]) and plateau[0] <= 0.3 and ts[p0 - 1] == plateau[0], "pauses at the first step at or below plateau_at * t_start"
    assert ts[p0 - 2] > 0.3
    assert np.all(np.diff(ts[p0 + 5 : 4 + 15]) < 0) and ts[4 + 14] == 0 and np.all(ts[-3:] == 0)
    without, _ = build_schedule(0.6, 4, 10, 3)
    assert np.array_equal(np.delete(ts, np.arange(p0, p0 + 5)), without), "the plateau only inserts steps"


def test_plateau_needs_a_time():
    with pytest.raises(ValueError):
        build_schedule(0.6, 4, 10, 3, plateau_at=None, plateau_steps=5)
