"""Score a results-style est jsonl (id/est_intervals/est_labels) against the
shared benchmark references (same refs/fits/metrics as every other run).

  python -m phraser.eval.score_est_jsonl --dataset salami --est-jsonl X --out Y
"""
import argparse
import json

import numpy as np

from phraser.eval.metrics import segment_scores
from phraser.eval.score_songformer_retrained import iter_refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["salami", "harmonix", "tuney"], required=True)
    ap.add_argument("--est-jsonl", required=True)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    want = None
    if args.ids:
        want = set(l.strip() for l in open(args.ids) if l.strip())
    ests = {}
    for line in open(args.est_jsonl):
        d = json.loads(line)
        if "est_intervals" in d:
            ests[d["id"]] = (np.asarray(d["est_intervals"], float), d.get("est_labels"))

    agg, n, missing = {}, 0, 0
    with open(args.out, "w") as fout:
        for tid, ref_int, ref_lab, fit in iter_refs(args.dataset):
            if want is not None and tid not in want:
                continue
            if tid not in ests:
                missing += 1
                continue
            est, labs = ests[tid]
            if labs is None:
                labs = ["inst"] * len(est)
            if fit is not None:
                rate, shift = fit
                est = np.clip(rate * est + shift, 0.0, None)
            row = {"id": tid}
            try:
                row.update(segment_scores(ref_int, ref_lab, est, labs))
            except Exception as e:
                row["error"] = repr(e)[:200]
            fout.write(json.dumps(row) + "\n")
            n += 1
            for k, v in row.items():
                if isinstance(v, (int, float)):
                    agg.setdefault(k, []).append(v)
    print("FINAL", n, "missing", missing,
          {k: round(float(np.mean(v)), 3) for k, v in agg.items()})


if __name__ == "__main__":
    main()
