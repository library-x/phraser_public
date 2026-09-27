"""Score any system's estimates against SongFormBench-HX-200 (their audio+labels).

SongFormBench labels are re-aligned to the bench's own audio, so estimates made
on that audio need no offset fits. Accepts either a SongFormer infer.py output
dir (per-track [{"label","start","end"}] JSON) or a results-style JSONL with
est_intervals/est_labels. Restrict with --ids (e.g. the 96-track subset
untouched by our Harmonix training).

Run:
  python -m phraser.eval.score_bhx --infer-dir <dir> [--ids splits/bhx_clean.txt] --out <jsonl>
"""
import argparse
import json
import os

import numpy as np

from phraser.eval.metrics import segment_scores
from phraser.eval.baseline_songformer import map_label
from phraser.paths import SONGFORMER

BENCH = f"{SONGFORMER}/songformbench/data/labels/HarmonixSet"


def load_refs():
    refs = {}
    for f in sorted(os.listdir(BENCH)):
        tid = f[4:-4]  # strip BHX_ / .txt
        rows = []
        for line in open(os.path.join(BENCH, f)):
            p = line.split()
            if len(p) >= 2:
                rows.append((float(p[0]), p[1]))
        if len(rows) < 2 or rows[-1][1] != "end":
            continue
        ints = np.array([[rows[i][0], rows[i + 1][0]] for i in range(len(rows) - 1)])
        labs = [map_label(r[1]) for r in rows[:-1]]
        refs[tid] = (ints, labs)
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer-dir", help="SongFormer infer.py output dir (BHX_<id>.json or <id>.json)")
    ap.add_argument("--est-jsonl", help="results-style jsonl with id/est_intervals/est_labels")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert bool(args.infer_dir) != bool(args.est_jsonl), "exactly one input source"

    refs = load_refs()
    want = None
    if args.ids:
        want = set(l.strip() for l in open(args.ids) if l.strip())

    ests = {}
    if args.est_jsonl:
        for line in open(args.est_jsonl):
            d = json.loads(line)
            if "est_intervals" in d:
                ests[d["id"]] = (np.asarray(d["est_intervals"], float), d.get("est_labels"))
    else:
        for f in os.listdir(args.infer_dir):
            if not f.endswith(".json"):
                continue
            tid = f[:-5]
            if tid.startswith("BHX_"):
                tid = tid[4:]
            segs = json.load(open(os.path.join(args.infer_dir, f)))
            ests[tid] = (np.array([[float(s["start"]), float(s["end"])] for s in segs]),
                         [map_label(str(s["label"])) for s in segs])

    agg, n, missing = {}, 0, 0
    with open(args.out, "w") as fout:
        for tid, (ref_int, ref_lab) in sorted(refs.items()):
            if want is not None and tid not in want:
                continue
            if tid not in ests:
                missing += 1
                continue
            est, labs = ests[tid]
            if labs is None:
                labs = ["inst"] * len(est)
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
