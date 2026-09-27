"""All-In-One (Kim & Nam 2023) baseline on the same audio + refs + metrics.

Run inside the `allin1` conda env (which has its own torch/natten). Only needs
mir_eval + numpy from this repo's eval code — imported via PYTHONPATH.

  python -m phraser.eval.baseline_allin1 --dataset salami \
      --root $PHRASER_DATA/benchmarks/salami-data-public \
      --audio-dir $PHRASER_DATA/benchmarks/salami_audio \
      --out $PHRASER_DATA/benchmarks/results_salami_allin1.jsonl
"""
import argparse
import json

import numpy as np

from phraser.eval import datasets as bench_datasets
from phraser.eval.datasets import map_label
from phraser.eval.metrics import segment_scores, beat_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["harmonix", "salami"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--demix-dir", default="./demix")
    ap.add_argument("--spec-dir", default="./spec")
    args = ap.parse_args()

    import allin1

    tracks = (bench_datasets.load_harmonix if args.dataset == "harmonix" else bench_datasets.load_salami)(
        args.root, audio_dir=args.audio_dir)
    with_audio = [t for t in tracks if t["audio"]]
    print(f"{args.dataset}: {len(with_audio)} with audio", flush=True)
    if args.limit:
        with_audio = with_audio[:args.limit]

    agg, n = {}, 0
    with open(args.out, "w") as fout:
        for t in with_audio:
            try:
                res = allin1.analyze(t["audio"], device="cuda", include_activations=False,
                                     demix_dir=args.demix_dir, spec_dir=args.spec_dir)
                est_intervals = np.array([[s.start, s.end] for s in res.segments])
                est_labels = [map_label(s.label) for s in res.segments]
                est_beats = np.asarray(res.beats, dtype=float)
                # Same YouTube-offset correction as the phraser run, estimated from
                # All-In-One's own predicted beat times (mode of pairwise time deltas).
                offset = 0.0
                if len(t["ref_beats"]) > 10 and len(est_beats) > 10:
                    d = (est_beats[:, None] - np.asarray(t["ref_beats"])[None, :]).ravel()
                    d = d[np.abs(d) <= 30.0]
                    if len(d):
                        hist, edges = np.histogram(d, bins=np.arange(-30.0, 30.01, 0.02))
                        offset = float((edges[hist.argmax()] + edges[hist.argmax() + 1]) / 2)
                if offset != 0.0:
                    est_intervals = np.clip(est_intervals - offset, 0.0, None)
                    est_beats = est_beats - offset
                    est_beats = est_beats[est_beats > 0]
                row = {"id": t["id"], "offset": round(offset, 2),
                       "est_intervals": np.round(est_intervals, 2).tolist(),
                       "est_labels": est_labels,
                       "est_beats": np.round(est_beats, 2).tolist()}
                try:
                    row.update(segment_scores(t["ref_intervals"], t["ref_labels"],
                                              est_intervals, est_labels))
                    if len(t["ref_beats"]) > 1 and len(est_beats) > 1:
                        row.update(beat_scores(t["ref_beats"], est_beats))
                except Exception as e:
                    row["error"] = repr(e)[:300]
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                n += 1
                for k, v in row.items():
                    if isinstance(v, (int, float)):
                        agg.setdefault(k, []).append(v)
                if n % 10 == 0:
                    print(n, {k: round(float(np.mean(v)), 3) for k, v in agg.items()}, flush=True)
            except Exception as e:
                fout.write(json.dumps({"id": t["id"], "error": repr(e)[:300]}) + "\n")
    print("FINAL", n, {k: round(float(np.mean(v)), 3) for k, v in agg.items()})


if __name__ == "__main__":
    main()
