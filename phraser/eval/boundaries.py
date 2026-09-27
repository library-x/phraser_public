"""Model logits -> (intervals, labels) and beat times.

Post-processing per the design doc: boundaries are predicted at 1 Hz on the
mixture channel, then snapped to the strongest predicted onset within a
±1 s neighbourhood (100 Hz). Vocal channel outputs are never consumed.
"""
import numpy as np

SECTION_LABELS = ["silence", "intro", "outro", "bridge", "verse", "chorus"]
HZ_COARSE = 1
HZ_FINE = 100


def pick_boundaries(split_prob, threshold=0.3, min_gap_s=4.0):
    """Local maxima of the 1 Hz mixture split probabilities above threshold."""
    split_prob = np.asarray(split_prob, dtype=float)
    cand = []
    for t in range(1, len(split_prob) - 1):
        if split_prob[t] >= threshold and split_prob[t] >= split_prob[t - 1] and split_prob[t] >= split_prob[t + 1]:
            cand.append((split_prob[t], t))
    cand.sort(reverse=True)
    chosen = []
    for p, t in cand:
        if all(abs(t - c) >= min_gap_s for c in chosen):
            chosen.append(t)
    return sorted(chosen)


def snap_to_onsets(bound_s, onset_prob, radius_s=1.0):
    """Snap 1 Hz boundaries (seconds) to the strongest 100 Hz onset nearby."""
    onset_prob = np.asarray(onset_prob, dtype=float)
    snapped = []
    for b in bound_s:
        c = int(b * HZ_FINE)
        lo, hi = max(0, c - int(radius_s * HZ_FINE)), min(len(onset_prob), c + int(radius_s * HZ_FINE) + 1)
        snapped.append((lo + int(np.argmax(onset_prob[lo:hi]))) / HZ_FINE if hi > lo else float(b))
    return snapped


def to_segments(bound_s, type_prob, duration_s):
    """Boundaries -> intervals + majority section label per interval (1 Hz type probs [T,6])."""
    type_prob = np.asarray(type_prob, dtype=float)
    if type_prob.ndim != 2 or type_prob.shape[1] != len(SECTION_LABELS):
        raise ValueError(
            f"type_prob must be [T, {len(SECTION_LABELS)}], got {type_prob.shape}"
            " (time-major; a transposed input would mislabel silently)")
    inner = sorted(float(b) for b in bound_s if 0.0 < b < duration_s)
    deduped = []
    for b in inner:  # onset-snapping can map two coarse boundaries to the same instant
        if not deduped or b - deduped[-1] >= 0.5:
            deduped.append(b)
    edges = [0.0] + deduped + [float(duration_s)]
    intervals, labels = [], []
    for s, e in zip(edges[:-1], edges[1:]):
        if e - s < 1e-3:
            continue
        i0, i1 = int(s * HZ_COARSE), max(int(e * HZ_COARSE), int(s * HZ_COARSE) + 1)
        labels.append(SECTION_LABELS[int(type_prob[i0:i1].mean(axis=0).argmax())])
        intervals.append([s, e])
    return np.array(intervals), labels


def beats_from_logits(beat_prob, threshold=0.5):
    """Predicted beat times in seconds via the periodic-grid reconstruction if available,
    else simple peak picking at 100 Hz."""
    try:
        from phraser.utils.utils import robust_periodic_reconstruction
        grid = np.asarray(robust_periodic_reconstruction(np.asarray(beat_prob)))
        return np.flatnonzero(grid) / HZ_FINE
    except Exception:
        p = np.asarray(beat_prob, dtype=float)
        idx = [t for t in range(1, len(p) - 1) if p[t] >= threshold and p[t] >= p[t - 1] and p[t] >= p[t + 1]]
        return np.array(idx) / HZ_FINE


def predict_track(model, embeddings, duration_s, torch=None):
    """embeddings: torch tensor [1, 4, T@25Hz, 1024] (vocals, other, bass, drums).
    Returns (intervals, labels, beat_times)."""
    import torch as _torch
    with _torch.no_grad():
        (split_l, _silent, sect_l), _, (beat_l, onset_l) = model.forward_encoded(embeddings)
    seg_prob = _torch.sigmoid(split_l[0, -1].squeeze(-1)).cpu().numpy()      # [T] 1 Hz mixture channel
    type_prob = _torch.softmax(sect_l[0], dim=-1).cpu().numpy()              # [T,6]
    onset_prob = _torch.sigmoid(onset_l[0].squeeze(-1)).cpu().numpy()        # [100T]
    beat_prob = _torch.sigmoid(beat_l[0].squeeze(-1)).cpu().numpy()
    bounds = snap_to_onsets(pick_boundaries(seg_prob), onset_prob)
    intervals, labels = to_segments(bounds, type_prob, duration_s)
    return intervals, labels, beats_from_logits(beat_prob), beat_prob


def estimate_offset(est_beats, ref_beats, max_shift_s=30.0, bin_s=0.02):
    """Constant audio-vs-annotation offset (YouTube-sourced benchmarks).

    Mode of pairwise (est_beat - ref_beat) time deltas. Positive means audio
    events occur later than the annotation timeline; subtract from estimates
    before scoring. NOTE: activation-correlation estimation was abandoned —
    beat grids are periodic, so that objective has near-equal maxima at
    beat-multiple lags and picks wrong ones; the pairwise-delta histogram
    concentrates mass at the true shift.
    """
    est = np.asarray(est_beats, dtype=float)
    ref = np.asarray(ref_beats, dtype=float)
    if len(ref) < 10 or len(est) < 10:
        return 0.0
    d = (est[:, None] - ref[None, :]).ravel()
    d = d[np.abs(d) <= max_shift_s]
    if not len(d):
        return 0.0
    hist, edges = np.histogram(d, bins=np.arange(-max_shift_s, max_shift_s + bin_s, bin_s))
    i = int(hist.argmax())
    return float((edges[i] + edges[i + 1]) / 2)
