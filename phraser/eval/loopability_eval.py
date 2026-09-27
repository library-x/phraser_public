"""Direct loopability evaluation from prediction dumps (CPU).

Metrics (tolerance 70 ms, matching beat-eval convention):
  on_beat      fraction of predicted interior boundaries within tol of an
               annotated beat
  on_downbeat  same, against downbeats (bar starts)
  loopable     fraction of predicted segments with BOTH endpoints within tol
               of downbeats (start on a downbeat, integer number of bars)
  chance       expected on_downbeat under uniform placement (2*tol*density)

Datasets/models:
  harmonix: phraser dumps (results_harmonix_full.jsonl, offset-aligned) and
            allin1 dumps (results_harmonix_allin1.jsonl); grid from
            beats_and_downbeats files (col2==1 -> downbeat).
  tuney:    phraser dumps (results_tuney_phraser_est.jsonl, from
            dump_tuney_est.py) and allin1 dumps (eval_allin1_tuney_e68.jsonl);
            grid reconstructed from bpm + robust bar offset, identical to the
            training-label pipeline.
"""
import glob
import json
import os

import numpy as np
from phraser.paths import BENCH, CORPUS, TUNEY_ALLIN1

BASE = BENCH
TOL = 0.07


def near(times, grid, tol=TOL):
    if len(grid) == 0 or len(times) == 0:
        return np.zeros(len(times), dtype=bool)
    grid = np.sort(np.asarray(grid, dtype=float))
    idx = np.searchsorted(grid, times)
    best = np.full(len(times), np.inf)
    for off in (-1, 0):
        j = np.clip(idx + off, 0, len(grid) - 1)
        best = np.minimum(best, np.abs(np.asarray(times) - grid[j]))
    return best <= tol


def track_metrics(est_intervals, beats, downbeats):
    est = np.asarray(est_intervals, dtype=float)
    if len(est) == 0:
        return None
    end = est[:, 1].max()
    interior = sorted({round(float(x), 3) for x in est.ravel() if 1e-3 < x < end - 1e-3})
    interior = np.array(interior)
    out = {}
    out["n_bounds"] = len(interior)
    out["on_beat"] = float(near(interior, beats).mean()) if len(interior) else None
    out["on_downbeat"] = float(near(interior, downbeats).mean()) if len(interior) else None
    both = near(est[:, 0], downbeats) & near(est[:, 1], downbeats)
    out["loopable"] = float(both.mean())
    dens = len(downbeats) / max(end, 1e-6)
    out["chance"] = min(1.0, 2 * TOL * dens)
    return out


def harmonix_grid(tid):
    beats, downs = [], []
    p = f"{BASE}/harmonixset/dataset/beats_and_downbeats/{tid}.txt"
    if not os.path.exists(p):
        return None, None
    for line in open(p):
        parts = line.split()
        if len(parts) >= 2:
            t = float(parts[0])
            beats.append(t)
            if parts[1] == "1":
                downs.append(t)
    return beats, downs


def tuney_grid(tid):
    import sys
    from phraser.utils.data_loader import find_robust_onsets
    jps = glob.glob(f"{CORPUS}/*/parsed/{tid}.json")
    if not jps:
        return None, None
    d = json.load(open(jps[0]))
    bpm = float(d["bpm"])
    bar_len, beat_len = 240.0 / bpm, 60.0 / bpm
    segs = d["segmnets"]
    dur = max(float(r["end"]) for r in segs) * bar_len
    ends = np.array([x["end"] for v in d["stem_elements"].values() for x in v], dtype=float)
    ob = find_robust_onsets(ends, length=1, tolerance=0.1)
    bar_off = float(ob[0] % 1)
    beat_off = bar_off % 0.25
    beats = [(k + beat_off) * beat_len for k in range(int(dur / beat_len) + 1)]
    downs = [(m + bar_off) * bar_len for m in range(int(dur / bar_len) + 1)]
    return beats, downs


def run(name, dump_path, grid_fn):
    aggs = {}
    n = 0
    for line in open(dump_path):
        r = json.loads(line)
        if "est_intervals" not in r:
            continue
        beats, downs = grid_fn(r["id"])
        if not beats:
            continue
        m = track_metrics(r["est_intervals"], beats, downs)
        if m is None:
            continue
        n += 1
        for k, v in m.items():
            if isinstance(v, (int, float)):
                aggs.setdefault(k, []).append(v)
    print(f"{name}: n={n} " + " ".join(f"{k}={np.mean(v):.3f}" for k, v in aggs.items()))


if __name__ == "__main__":
    run("phraser/harmonix", f"{BASE}/results_harmonix_full.jsonl", harmonix_grid)
    run("allin1/harmonix ", f"{BASE}/results_harmonix_allin1.jsonl", harmonix_grid)
    run("phraser/tuney   ", f"{BASE}/results_tuney_phraser_est.jsonl", tuney_grid)
    run("allin1/tuney    ", f"{TUNEY_ALLIN1}/eval_allin1_tuney_e68.jsonl", tuney_grid)
