"""Re-score the phraser Harmonix run from raw dumps with corrected offsets.

The original run's offsets came from the (flawed) activation-correlation
estimator. This undoes them and re-estimates via the pairwise-delta histogram
on the dumped beat times. CPU-only.
"""
import json

import numpy as np

from phraser.eval import datasets as bench_datasets
from phraser.eval.boundaries import estimate_offset
from phraser.eval.metrics import segment_scores, beat_scores
from phraser.paths import BENCH

BASE = BENCH


def main():
    tracks = {t["id"]: t for t in bench_datasets.load_harmonix(f"{BASE}/harmonixset")}
    rows = [json.loads(l) for l in open(f"{BASE}/results_harmonix_full.jsonl")]
    agg, n = {}, 0
    with open(f"{BASE}/results_harmonix_rescored.jsonl", "w") as fout:
        for r in rows:
            if "est_intervals" not in r or r["id"] not in tracks:
                continue
            t = tracks[r["id"]]
            old_off = r.get("offset", 0.0)
            est_int = np.asarray(r["est_intervals"], dtype=float) + old_off  # back to audio frame
            est_beats = np.asarray(r["est_beats"], dtype=float) + old_off
            offset = estimate_offset(est_beats, t["ref_beats"])
            est_int = np.clip(est_int - offset, 0.0, None)
            est_beats = est_beats - offset
            est_beats = est_beats[est_beats > 0]
            try:
                row = {"id": r["id"], "offset": round(offset, 2),
                       **segment_scores(t["ref_intervals"], t["ref_labels"], est_int, r["est_labels"])}
                if len(t["ref_beats"]) > 1 and len(est_beats) > 1:
                    row.update(beat_scores(t["ref_beats"], est_beats))
            except Exception as e:
                row = {"id": r["id"], "error": repr(e)[:200]}
            fout.write(json.dumps(row) + "\n")
            n += 1
            for k, v in row.items():
                if isinstance(v, (int, float)):
                    agg.setdefault(k, []).append(v)
    print(f"RESCORED n={n}")
    print("RESULT", {k: round(float(np.mean(v)), 3) for k, v in agg.items()})


if __name__ == "__main__":
    main()
