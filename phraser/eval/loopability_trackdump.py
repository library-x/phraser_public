"""Loopability + stem-loop recovery for any track-level est dump on Tuney test.

Reuses the exact conventions of loopability_eval (grid from bpm + robust bar
offset; loopable = both segment endpoints within tol of downbeats) and
stem_loopability_eval (element recovered iff BOTH endpoints matched by a
predicted boundary within 5 ms; track-level systems replicate their mixture
boundaries to every stem).

  python -m phraser.eval.loopability_trackdump --dump results_tuney_X.jsonl
"""
import argparse
import glob
import json
import os

import numpy as np

from phraser.eval.loopability_eval import tuney_grid, near
from phraser.eval.stem_loopability_eval import match, stem_truth, score_stem
from phraser.paths import corpus_splits

TOLS = (0.005, 0.010, 0.030, 0.070)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    args = ap.parse_args()

    # bpm/stem_elements ground truth per test id
    meta = {}
    for ds in corpus_splits():
        b = os.path.dirname(ds)
        for tid in json.load(open(ds))["test"]:
            jp = f"{b}/parsed/{tid}.json"
            if os.path.exists(jp):
                meta[tid] = jp

    loop = {t: [] for t in TOLS}
    el_rec, n = [], 0
    for line in open(args.dump):
        r = json.loads(line)
        if "est_intervals" not in r or r["id"] not in meta:
            continue
        beats, downs = tuney_grid(r["id"])
        if not beats:
            continue
        iv = [x for x in r["est_intervals"] if x[1] - x[0] > 1e-6]
        if not iv:
            continue
        n += 1
        for tol in TOLS:
            ok = [bool(near([a], downs, tol)[0] and near([b], downs, tol)[0])
                  for a, b in iv]
            loop[tol].append(float(np.mean(ok)))
        d = json.load(open(meta[r["id"]]))
        bar_len = 240.0 / float(d["bpm"])
        bounds = sorted({round(float(x), 3) for pair in r["est_intervals"] for x in pair})
        recs = []
        for name, els in stem_truth(d, bar_len).items():
            s = score_stem(bounds, els)
            if s is not None:
                recs.append(s["el_recall"])
        if recs:
            el_rec.append(float(np.mean(recs)))

    print(f"n={n}")
    for tol in TOLS:
        print(f"loopable@{int(tol*1000)}ms: {np.mean(loop[tol]):.3f}")
    print(f"stem el_recall@5ms (replicated): {np.mean(el_rec):.3f}")


if __name__ == "__main__":
    main()
