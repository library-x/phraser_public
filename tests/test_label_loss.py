"""Tests for the class-axis label loss.

Regression context: the original training code passed [B, T, C] logits and
one-hot float targets straight into F.cross_entropy, whose class axis is
dim 1 --- so the softmax ran over TIME. That objective is invariant to a
constant offset on a class's entire timeline, leaving inter-class
calibration untrained (the 'intro blob'), and its magnitude scaled with
the number of labeled frames (~n_c x CE), silently inflating the label
term ~100x past its configured weight.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from phraser.utils.losses import label_cross_entropy

C = 6


def one_hot(idx):
    return F.one_hot(idx, C).float()


def test_perfect_prediction_near_zero():
    tgt = torch.randint(0, C, (2, 40))
    oh = one_hot(tgt)
    assert float(label_cross_entropy(oh * 20.0, oh)) < 1e-3


def test_random_logits_score_log_c_regardless_of_length():
    # The bugged loss scaled with the number of labeled frames (~180 per
    # track); the corrected loss is a per-frame mean, ~log(C) at init.
    torch.manual_seed(0)
    losses = []
    for T in (30, 300):
        logits = torch.randn(1, T, C) * 0.01
        oh = one_hot(torch.randint(0, C, (1, T)))
        losses.append(float(label_cross_entropy(logits, oh)))
    for loss in losses:
        assert abs(loss - math.log(C)) < 0.05
    assert abs(losses[0] - losses[1]) < 0.05


def test_class_offset_is_punished_regression():
    # A class row drifting upward must increase the loss --- this is the
    # untrained degree of freedom that produced the prior blob.
    torch.manual_seed(1)
    tgt = torch.randint(1, C, (1, 60))  # class 0 never annotated
    oh = one_hot(tgt)
    logits = oh * 4.0
    shifted = logits.clone()
    shifted[..., 0] += 10.0
    assert float(label_cross_entropy(shifted, oh)) > \
        float(label_cross_entropy(logits, oh)) + 1.0


def test_bugged_call_was_offset_invariant_canary():
    # Documents the failure the wrapper prevents: the raw [B, T, C] call
    # normalizes over dim 1 (time), so the same class-row offset changes
    # nothing. If this canary ever fails, torch semantics changed.
    torch.manual_seed(1)
    tgt = torch.randint(1, C, (1, 60))
    oh = one_hot(tgt)
    logits = oh * 4.0
    shifted = logits.clone()
    shifted[..., 0] += 10.0
    assert float(F.cross_entropy(shifted, oh)) == \
        pytest.approx(float(F.cross_entropy(logits, oh)))


def test_gradient_pushes_true_class_up():
    logits = torch.zeros(1, 10, C, requires_grad=True)
    oh = one_hot(torch.full((1, 10), 3))
    label_cross_entropy(logits, oh).backward()
    g = logits.grad
    assert torch.all(g[..., 3] < 0)
    other = torch.ones(C, dtype=torch.bool)
    other[3] = False
    assert torch.all(g[..., other] > 0)


def test_padding_rows_ignored():
    torch.manual_seed(2)
    tgt = torch.randint(0, C, (1, 20))
    oh = one_hot(tgt)
    padded = torch.cat([oh, torch.zeros(1, 15, C)], dim=1)
    logits = torch.randn(1, 35, C)
    a = float(label_cross_entropy(logits, padded))
    b = float(label_cross_entropy(logits[:, :20], oh))
    assert a == pytest.approx(b)
    noisy = logits.clone()
    noisy[:, 20:] += 100.0  # padded-frame logits must not matter
    assert float(label_cross_entropy(noisy, padded)) == pytest.approx(a)


def test_shape_guards():
    with pytest.raises(ValueError):
        label_cross_entropy(torch.zeros(1, C, 240), torch.zeros(1, C, 240))
    with pytest.raises(ValueError):
        label_cross_entropy(torch.zeros(1, 10, C), torch.zeros(1, 11, C))


def test_matches_index_target_cross_entropy():
    torch.manual_seed(3)
    logits = torch.randn(2, 50, C)
    tgt = torch.randint(0, C, (2, 50))
    a = float(label_cross_entropy(logits, one_hot(tgt)))
    b = float(F.cross_entropy(logits.reshape(-1, C), tgt.reshape(-1)))
    assert a == pytest.approx(b, rel=1e-5)
