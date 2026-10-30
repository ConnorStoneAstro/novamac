"""The diffusion-time schedule, with and without mid-anneal plateaus."""

import numpy as np
import pytest

from novamac.sampler import build_schedule, lagging_chains


def test_schedule_without_plateau():
    ts, starts = build_schedule(0.6, 4, 10, 3)
    assert len(starts) == 0 and len(ts) == 17
    assert np.all(ts[:4] == 0.6) and np.all(ts[-3:] == 0) and np.all(np.diff(ts[4:14]) < 0)
    assert ts[13] == 0


def test_schedule_with_plateau():
    ts, starts = build_schedule(0.6, 4, 10, 3, plateau_at=0.5, plateau_steps=5)
    assert len(ts) == 4 + 10 + 5 + 3 and len(starts) == 1
    p0 = int(starts[0])
    plateau = ts[p0 : p0 + 5]
    assert np.all(plateau == plateau[0]) and plateau[0] <= 0.3 and ts[p0 - 1] == plateau[0], "pauses at the first step at or below plateau_at * t_start"
    assert ts[p0 - 2] > 0.3
    assert np.all(np.diff(ts[p0 + 5 : 4 + 15]) < 0) and ts[4 + 14] == 0 and np.all(ts[-3:] == 0)
    without, _ = build_schedule(0.6, 4, 10, 3)
    assert np.array_equal(np.delete(ts, np.arange(p0, p0 + 5)), without), "the plateau only inserts steps"


def test_schedule_with_several_plateaus():
    fractions = [0.2, 0.8, 0.5]  # any order
    ts, starts = build_schedule(0.6, 4, 10, 3, plateau_at=fractions, plateau_steps=5)
    assert len(ts) == 4 + 10 + 3 * 5 + 3 and len(starts) == 3
    assert np.all(np.diff(starts) > 0), "plateaus in anneal order (highest t first)"
    for p0, frac in zip(starts, sorted(fractions, reverse=True)):
        plateau = ts[p0 : p0 + 5]
        assert np.all(plateau == plateau[0]) and plateau[0] <= frac * 0.6 + 1e-12 and ts[p0 - 1] == plateau[0]
        assert ts[p0 - 2] > frac * 0.6
    without, _ = build_schedule(0.6, 4, 10, 3)
    inserted = np.concatenate([np.arange(p0, p0 + 5) for p0 in starts])
    assert np.array_equal(np.delete(ts, inserted), without), "the plateaus only insert steps"


def test_plateau_needs_a_time():
    with pytest.raises(ValueError):
        build_schedule(0.6, 4, 10, 3, plateau_at=None, plateau_steps=5)
    with pytest.raises(ValueError):
        build_schedule(0.6, 4, 10, 3, plateau_at=[0.5, 1.5], plateau_steps=5)


def test_lagging_chains_restart_after_persisting():
    counts = np.zeros(4, dtype=int)
    ll = np.array([-100.0, -3000.0, -150.0, -2200.0])  # chains 1 and 3 more than 2000 behind chain 0
    lagging, counts = lagging_chains(ll, counts, gap=2000.0, persist=2)
    assert not lagging.any() and list(counts) == [0, 1, 0, 1], "one chunk behind is not enough"
    lagging, counts = lagging_chains(ll, counts, gap=2000.0, persist=2)
    assert list(lagging) == [False, True, False, True] and list(counts) == [0, 0, 0, 0], "second chunk: restart, counts reset"
    ll = np.array([-100.0, -3000.0, -150.0, -1000.0])
    lagging, counts = lagging_chains(ll, counts, gap=2000.0, persist=2)
    assert not lagging.any() and list(counts) == [0, 1, 0, 0], "a chain back within the gap is cleared"
    # the top chain never lags, however far the others are
    lagging, counts = lagging_chains(np.array([-1e5, -1e5 - 5000]), np.array([5, 5]), gap=2000.0, persist=2)
    assert list(lagging) == [False, True]
