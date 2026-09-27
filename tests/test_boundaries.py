import numpy as np
import pytest

from phraser.eval.boundaries import (
    SECTION_LABELS, pick_boundaries, snap_to_onsets, to_segments, estimate_offset,
)
from phraser.eval.sweep_postproc import merge_same


def test_pick_boundaries_finds_separated_peaks():
    p = np.zeros(100)
    p[20] = 0.9
    p[60] = 0.8
    assert pick_boundaries(p, threshold=0.3, min_gap_s=4.0) == [20, 60]


def test_pick_boundaries_min_gap_keeps_strongest():
    p = np.zeros(100)
    p[30] = 0.9
    p[32] = 0.7  # within min gap of the stronger peak
    assert pick_boundaries(p, threshold=0.3, min_gap_s=4.0) == [30]


def test_pick_boundaries_threshold():
    p = np.zeros(100)
    p[50] = 0.2
    assert pick_boundaries(p, threshold=0.3) == []


def test_snap_to_onsets_moves_to_strongest_neighbor():
    onset = np.zeros(3000)  # 30 s at 100 Hz
    onset[1030] = 0.9       # 10.30 s
    snapped = snap_to_onsets([10], onset, radius_s=1.0)
    assert snapped[0] == pytest.approx(10.30)


def test_to_segments_dedupes_close_boundaries_regression():
    # Two coarse boundaries snapping to the same onset created zero-length
    # intervals that crashed mir_eval.
    T = 60
    type_prob = np.zeros((T, len(SECTION_LABELS)))
    type_prob[:, 4] = 1.0  # 'verse'
    intervals, labels = to_segments([20.0, 20.2, 40.0], type_prob, float(T))
    assert (intervals[:, 1] - intervals[:, 0] > 0).all()
    assert len(intervals) == 3  # 0-20, 20-40, 40-60
    assert set(labels) == {"verse"}


def test_estimate_offset_recovers_shift_on_jittered_beats():
    rng = np.random.default_rng(0)
    ref = np.cumsum(rng.uniform(0.4, 0.6, size=200))  # aperiodic beat times
    est = ref + 3.37
    off = estimate_offset(est, ref)
    assert off == pytest.approx(3.37, abs=0.03)


def test_estimate_offset_periodic_grid_no_multiple_lock_regression():
    # The activation-correlation estimator locked onto beat-multiple lags on
    # periodic grids; the pairwise-delta mode must pick the true shift.
    ref = np.arange(0.0, 120.0, 0.5)  # strictly periodic
    est = ref + 3.0
    off = estimate_offset(est, ref)
    assert off == pytest.approx(3.0, abs=0.03)


def test_estimate_offset_too_few_beats_returns_zero():
    assert estimate_offset(np.arange(5), np.arange(5)) == 0.0


def test_merge_same_adjacent_labels():
    intervals = np.array([[0.0, 10.0], [10.0, 20.0], [20.0, 30.0]])
    merged_i, merged_l = merge_same(intervals, ["a", "a", "b"])
    assert merged_l == ["a", "b"]
    assert merged_i[0][1] == 20.0
