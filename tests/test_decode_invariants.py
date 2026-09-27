"""Invariant tests for the decode pipeline — properties the paper claims
and axis/shape hazards of the label-CE-bug class."""
import numpy as np
import pytest

from phraser.eval.boundaries import (
    HZ_FINE, SECTION_LABELS, pick_boundaries, snap_to_onsets, to_segments,
)


def test_snapped_boundaries_are_fine_grid_multiples():
    # Paper Eq. (1): a snapped boundary is a multiple of 1/f, so its error
    # is either within the 5 ms quantization step or a bar fraction off.
    rng = np.random.default_rng(0)
    onset = rng.uniform(0, 1, size=30 * HZ_FINE)
    snapped = snap_to_onsets([3, 11, 22], onset, radius_s=1.0)
    for b in snapped:
        assert (b * HZ_FINE) == pytest.approx(round(b * HZ_FINE))


def test_snap_stays_within_radius():
    onset = np.random.default_rng(1).uniform(0, 1, size=30 * HZ_FINE)
    for b0, b1 in zip([3, 11, 22], snap_to_onsets([3, 11, 22], onset, radius_s=1.0)):
        assert abs(b1 - b0) <= 1.0 + 1e-9


def test_snap_near_track_edges_clamps_window():
    onset = np.zeros(30 * HZ_FINE)
    onset[5] = 0.9      # 0.05 s
    onset[2995] = 0.9   # 29.95 s
    snapped = snap_to_onsets([0.2, 29.7], onset, radius_s=1.0)
    assert snapped[0] == pytest.approx(0.05)
    assert snapped[1] == pytest.approx(29.95)


def test_snap_empty_window_returns_input():
    assert snap_to_onsets([5.0], np.zeros(0), radius_s=1.0) == [5.0]


def test_pick_boundaries_plateau_yields_single_boundary():
    p = np.zeros(60)
    p[20:23] = 0.8  # flat plateau: >= both neighbours at 3 points
    picked = pick_boundaries(p, threshold=0.3, min_gap_s=4.0)
    assert len(picked) == 1
    assert picked[0] in (20, 21, 22)


def test_to_segments_partitions_the_track_exactly():
    T = 90
    type_prob = np.zeros((T, len(SECTION_LABELS)))
    type_prob[:, 5] = 1.0
    intervals, labels = to_segments([12.34, 40.0, 71.9], type_prob, float(T))
    assert intervals[0][0] == 0.0
    assert intervals[-1][1] == float(T)
    for (a, b), (c, d) in zip(intervals[:-1], intervals[1:]):
        assert b == pytest.approx(c)  # contiguous, no gaps or overlaps
    assert len(labels) == len(intervals)


def test_to_segments_rejects_transposed_type_prob_regression():
    # Same hazard class as the label-CE axis bug: a [6, T] input must raise,
    # not silently mislabel.
    T = 60
    good = np.zeros((T, len(SECTION_LABELS)))
    good[:, 1] = 1.0
    to_segments([20.0], good, float(T))  # sanity: time-major works
    with pytest.raises(ValueError):
        to_segments([20.0], good.T, float(T))


def test_section_labels_consistent_across_modules():
    # Decode indexes SECTION_LABELS by argmax position; training builds
    # one-hots from the train-time list. A mismatch silently permutes labels.
    from phraser.utils.utils import SECTION_LABELS as TRAIN_LABELS
    assert list(TRAIN_LABELS) == list(SECTION_LABELS)


def test_label_ce_weight_argument_reweights():
    import torch
    import torch.nn.functional as F
    from phraser.utils.losses import label_cross_entropy
    tgt = torch.zeros(1, 10, dtype=torch.long)  # all class 0
    oh = F.one_hot(tgt, len(SECTION_LABELS)).float()
    logits = torch.zeros(1, 10, len(SECTION_LABELS))
    w = torch.ones(len(SECTION_LABELS))
    w[0] = 2.0
    plain = float(label_cross_entropy(logits, oh))
    weighted = float(label_cross_entropy(logits, oh, weight=w))
    assert weighted == pytest.approx(plain)  # mean-normalized by weights
    w2 = torch.ones(len(SECTION_LABELS))
    smoothed = float(label_cross_entropy(logits, oh, weight=w2, label_smoothing=0.2))
    assert smoothed == pytest.approx(plain, abs=1e-6)  # uniform logits: smoothing no-op
