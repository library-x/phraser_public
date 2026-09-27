import numpy as np
import pytest

from phraser.eval.metrics import segment_scores, beat_scores


def test_perfect_segmentation_scores_one():
    ref = np.array([[0.0, 10.0], [10.0, 20.0], [20.0, 30.0]])
    labs = ["verse", "chorus", "verse"]
    s = segment_scores(ref, labs, ref.copy(), list(labs))
    assert s["HR.5F"] == pytest.approx(1.0)
    assert s["HR3F"] == pytest.approx(1.0)
    assert s["PWF"] == pytest.approx(1.0)


def test_shifted_boundary_hits_only_wide_window():
    ref = np.array([[0.0, 10.0], [10.0, 20.0]])
    est = np.array([[0.0, 11.5], [11.5, 20.0]])
    labs = ["verse", "chorus"]
    s = segment_scores(ref, labs, est, list(labs))
    assert s["HR.5F"] == pytest.approx(0.0)
    assert s["HR3F"] == pytest.approx(1.0)


def test_ref_not_starting_at_zero_regression():
    # Harmonix refs often start > 0; mir_eval requires [0, t_max] spans.
    # This crashed 501/771 tracks before span normalization was added.
    ref = np.array([[4.2, 10.0], [10.0, 20.0]])
    est = np.array([[0.0, 10.0], [10.0, 20.0]])
    s = segment_scores(ref, ["verse", "chorus"], est, ["verse", "chorus"])
    assert 0.0 <= s["HR3F"] <= 1.0


def test_zero_length_ref_rows_dropped_regression():
    # SALAMI annotations can carry duplicate timestamps.
    ref = np.array([[0.0, 5.0], [5.0, 5.0], [5.0, 10.0]])
    est = np.array([[0.0, 5.0], [5.0, 10.0]])
    s = segment_scores(ref, ["verse", "chorus", "verse"], est, ["verse", "verse"])
    assert 0.0 <= s["PWF"] <= 1.0


def test_beat_scores_identical_is_one():
    beats = np.arange(1.0, 60.0, 0.5)
    assert beat_scores(beats, beats.copy())["beat_F"] == pytest.approx(1.0)


def test_beat_scores_offbeat_is_zero():
    beats = np.arange(1.0, 60.0, 0.5)
    assert beat_scores(beats, beats + 0.25)["beat_F"] == pytest.approx(0.0)
