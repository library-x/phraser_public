"""mir_eval wrappers: one call per track, one flat dict out."""
import numpy as np
import mir_eval


def segment_scores(ref_intervals, ref_labels, est_intervals, est_labels):
    """Boundary HR F1 at 0.5s/3s, pairwise F1, and label agreement.

    Intervals: np.ndarray [N,2] seconds; labels: list[str] length N.
    Est intervals are adjusted (trimmed/padded with 'silence') to the ref span,
    as mir_eval's pairwise/NCE metrics require identical time ranges.
    """
    ref_intervals = np.asarray(ref_intervals, dtype=float)
    est_intervals = np.asarray(est_intervals, dtype=float)
    ref_labels, est_labels = list(ref_labels), list(est_labels)
    keep = ref_intervals[:, 1] - ref_intervals[:, 0] > 1e-6  # some refs carry duplicate timestamps
    ref_intervals, ref_labels = ref_intervals[keep], [l for l, k in zip(ref_labels, keep) if k]
    keep = est_intervals[:, 1] - est_intervals[:, 0] > 1e-6
    est_intervals, est_labels = est_intervals[keep], [l for l, k in zip(est_labels, keep) if k]
    t_max = float(ref_intervals.max())
    # mir_eval requires both spans to be [0, t_max]; many refs (e.g. Harmonix) start > 0
    ref_intervals, ref_labels = mir_eval.util.adjust_intervals(
        ref_intervals, ref_labels, t_min=0.0, t_max=t_max,
        start_label="silence", end_label="silence")
    est_intervals, est_labels = mir_eval.util.adjust_intervals(
        est_intervals, est_labels, t_min=0.0, t_max=t_max,
        start_label="silence", end_label="silence")
    out = {}
    _, _, out["HR.5F"] = mir_eval.segment.detection(ref_intervals, est_intervals, window=0.5, trim=True)
    _, _, out["HR3F"] = mir_eval.segment.detection(ref_intervals, est_intervals, window=3.0, trim=True)
    p, r, f = mir_eval.segment.pairwise(ref_intervals, ref_labels, est_intervals, est_labels)
    out["PWF"] = f
    out["PWP"], out["PWR"] = p, r
    s_over, s_under, s_f = mir_eval.segment.nce(ref_intervals, ref_labels, est_intervals, est_labels)
    out["Sf"] = s_f
    return out


def beat_scores(ref_beats, est_beats):
    """Standard beat F-measure (70 ms window)."""
    ref_beats = mir_eval.beat.trim_beats(np.asarray(ref_beats, dtype=float))
    est_beats = mir_eval.beat.trim_beats(np.asarray(est_beats, dtype=float))
    return {"beat_F": mir_eval.beat.f_measure(ref_beats, est_beats)}
