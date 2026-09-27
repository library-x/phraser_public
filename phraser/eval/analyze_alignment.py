"""Harmonix alignment forensics + re-scoring study (CPU, from raw dumps).

For each track (both models' dumps): undo stored constant offset, refit
  (a) constant offset (pairwise-delta mode)
  (b) linear map t_ref ~= rate * t_audio + shift (iterated matched-pair LS)
then re-score segment detection under {constant, linear} x {trim True/False},
plus an alignment-quality gate (frac of ref beats matched within 70 ms).

Alignment is always fitted on ALLIN1's predicted beats (the stronger beat
tracker) and the same per-track map is applied to BOTH models' estimates, so
correction quality never favours either model.
"""
import json

import numpy as np
import mir_eval

from phraser.eval import datasets as bench_datasets
from phraser.paths import BENCH

BASE = BENCH


def match_pairs(est, ref, tol=0.35):
    i = np.searchsorted(est, ref)
    pairs = []
    for k, r in enumerate(ref):
        cands = []
        if i[k] < len(est):
            cands.append(est[i[k]])
        if i[k] > 0:
            cands.append(est[i[k] - 1])
        if not cands:
            continue
        e = min(cands, key=lambda x: abs(x - r))
        if abs(e - r) <= tol:
            pairs.append((e, r))
    return np.array(pairs)


def fit_linear(est_beats, ref_beats, const_offset):
    """Iteratively refine t_ref = rate * t_audio + shift starting from the constant fit."""
    rate, shift = 1.0, -const_offset  # t_ref = t_audio - offset initially
    for _ in range(4):
        mapped = rate * est_beats + shift
        pairs = match_pairs(np.sort(mapped), ref_beats)
        if len(pairs) < 20:
            return rate, shift, pairs
        # refit on audio-domain matched pairs: recover audio times of matched est
        est_audio = (pairs[:, 0] - shift) / rate
        A = np.vstack([est_audio, np.ones(len(est_audio))]).T
        sol, *_ = np.linalg.lstsq(A, pairs[:, 1], rcond=None)
        rate, shift = float(sol[0]), float(sol[1])
    mapped = rate * est_beats + shift
    pairs = match_pairs(np.sort(mapped), ref_beats)
    return rate, shift, pairs


def load_dumps(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if "est_intervals" in r:
            out[r["id"]] = r
    return out


def detection(ref_int, est_int, window, trim):
    try:
        _, _, f = mir_eval.segment.detection(ref_int, est_int, window=window, trim=trim)
        return f
    except Exception:
        return None


def main():
    tracks = {t["id"]: t for t in bench_datasets.load_harmonix(f"{BASE}/harmonixset")}
    dumps = {"phraser": load_dumps(f"{BASE}/results_harmonix_full.jsonl"),
             "allin1": load_dumps(f"{BASE}/results_harmonix_allin1.jsonl")}
    ids = sorted(set(dumps["phraser"]) & set(dumps["allin1"]) & set(tracks))
    print(f"common tracks: {len(ids)}")

    fits = {}
    gate_stats = []
    for tid in ids:
        t = tracks[tid]
        ra = dumps["allin1"][tid]
        est_audio = np.asarray(ra["est_beats"], dtype=float) + ra.get("offset", 0.0)
        if len(est_audio) < 20 or len(t["ref_beats"]) < 20:
            continue
        d = (est_audio[:, None] - np.asarray(t["ref_beats"])[None, :]).ravel()
        d = d[np.abs(d) <= 30]
        if not len(d):
            continue
        hist, edges = np.histogram(d, bins=np.arange(-30, 30.02, 0.02))
        const = float((edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2)
        rate, shift, pairs = fit_linear(est_audio, np.asarray(t["ref_beats"]), const)
        if not (0.95 < rate < 1.05):
            rate, shift = 1.0, -const
            pairs = match_pairs(np.sort(est_audio - const), np.asarray(t["ref_beats"]))
        frac70 = float((np.abs(pairs[:, 0] - pairs[:, 1]) <= 0.07).sum() / max(len(t["ref_beats"]), 1)) if len(pairs) else 0.0
        resid = float(np.std(pairs[:, 0] - pairs[:, 1])) if len(pairs) else None
        fits[tid] = {"const": const, "rate": rate, "shift": shift, "frac70": frac70, "resid": resid}
        gate_stats.append(frac70)

    gate_stats = np.array(gate_stats)
    print(f"alignment gate: frac of ref beats matched@70ms — median {np.median(gate_stats):.2f}; "
          f">=0.7 on {(gate_stats >= 0.7).mean():.0%} of tracks; rate!=1 beyond 0.1%: "
          f"{np.mean([abs(f['rate']-1) > 0.001 for f in fits.values()]):.0%}")

    for model in ["phraser", "allin1"]:
        for corr in ["const", "linear"]:
            for trim in [True, False]:
                for gated in [False, True]:
                    h5, h3 = [], []
                    for tid, f in fits.items():
                        if gated and f["frac70"] < 0.7:
                            continue
                        r = dumps[model][tid]
                        t = tracks[tid]
                        est_audio = np.asarray(r["est_intervals"], dtype=float) + r.get("offset", 0.0)
                        if corr == "const":
                            est = est_audio - f["const"]
                        else:
                            est = f["rate"] * est_audio + f["shift"]
                        est = np.clip(est, 0.0, None)
                        keep = est[:, 1] - est[:, 0] > 1e-6
                        est = est[keep]
                        if len(est) < 1:
                            continue
                        f5 = detection(t["ref_intervals"], est, 0.5, trim)
                        f3 = detection(t["ref_intervals"], est, 3.0, trim)
                        if f5 is not None:
                            h5.append(f5)
                        if f3 is not None:
                            h3.append(f3)
                    tag = f"{model:8s} {corr:6s} trim={str(trim):5s} gated={str(gated):5s}"
                    print(f"{tag} n={len(h5):3d}  HR.5F {np.mean(h5):.3f}  HR3F {np.mean(h3):.3f}")

    json.dump(fits, open(f"{BASE}/harmonix_alignment_fits.json", "w"))


if __name__ == "__main__":
    main()
