"""Score SongFormer infer.py outputs (retrained checkpoints) with the shared harness.

Reads the per-track JSON produced by SongFormer's own infer/infer.py
([{"label","start","end"}, ...], audio timeline), maps Harmonix estimates into
the annotation timeline with the shared per-track fits (t_ref = rate*t_audio +
shift, identical to every other system), and scores with
phraser.eval.metrics.segment_scores against the same references the benchmark
uses. Restrict with --ids to the held-out test list.

Run:
  python -m phraser.eval.score_songformer_retrained \
      --dataset harmonix --infer-dir $PHRASER_DATA/songformer_train/infer_hx_test \
      --ids $PHRASER_DATA/songformer_train/splits/harmonix_test.txt \
      --out $PHRASER_DATA/benchmarks/results_harmonix_songformer_retrained.jsonl
"""
import argparse
import glob
import json
import os

import numpy as np

from phraser.eval import datasets as bench_datasets
from phraser.eval.metrics import segment_scores
from phraser.eval.baseline_songformer import map_label
from phraser.paths import BENCH, corpus_splits

BASE = BENCH


def iter_refs(dataset):
    if dataset == "salami":
        for t in bench_datasets.load_salami(f"{BASE}/salami-data-public",
                                            audio_dir=f"{BASE}/salami_audio"):
            yield t["id"], np.asarray(t["ref_intervals"], float), t["ref_labels"], None
    elif dataset == "harmonix":
        fits = json.load(open(f"{BASE}/harmonix_alignment_fits.json"))
        for t in bench_datasets.load_harmonix(f"{BASE}/harmonixset",
                                              audio_dir=f"{BASE}/harmonix_audio"):
            if t["id"] in fits:
                f = fits[t["id"]]
                yield t["id"], np.asarray(t["ref_intervals"], float), t["ref_labels"], (f["rate"], f["shift"])
    else:
        for ds in corpus_splits():
            b = os.path.dirname(ds)
            for tid in json.load(open(ds))["test"]:
                jp = f"{b}/parsed/{tid}.json"
                if not os.path.exists(jp):
                    continue
                d = json.load(open(jp))
                bar = 240.0 / float(d["bpm"])
                segs = sorted(((float(r["start"]) * bar, float(r["end"]) * bar,
                                map_label(str(r["name"])))
                               for r in d["segmnets"]), key=lambda z: z[0])
                yield tid, np.array([[s, e] for s, e, _ in segs]), [l for _, _, l in segs], None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["salami", "harmonix", "tuney"], required=True)
    ap.add_argument("--infer-dir", required=True)
    ap.add_argument("--ids", default=None, help="optional txt with one id per line")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    want = None
    if args.ids:
        want = set(l.strip() for l in open(args.ids) if l.strip())

    agg, n, missing = {}, 0, 0
    with open(args.out, "w") as fout:
        for tid, ref_int, ref_lab, fit in iter_refs(args.dataset):
            if want is not None and tid not in want:
                continue
            jp = os.path.join(args.infer_dir, f"{tid}.json")
            if not os.path.exists(jp):
                missing += 1
                continue
            segs = json.load(open(jp))
            est = np.array([[float(s["start"]), float(s["end"])] for s in segs])
            labs = [map_label(str(s["label"])) for s in segs]
            if fit is not None:  # audio frame -> annotation frame
                rate, shift = fit
                est = np.clip(rate * est + shift, 0.0, None)
            row = {"id": tid,
                   "est_intervals": np.round(est, 3).tolist(),
                   "est_labels": labs}
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
