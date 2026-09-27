"""Post-processing sweep on dumped activations (CPU only).

Tunes boundary threshold, min-gap, onset-snapping and same-label merging on a
VALIDATION partition, then reports untuned vs tuned on the TEST partition.
Also (harmonix) compares grid-reconstruction beats vs madmom DBN decoding.

Partitions: harmonix -> harmonix_ft/split.json val/test; salami -> ids hashed
(md5 mod 4 == 0 -> val, else test), documented in the paper.
Run in allin1 env (madmom): python -u -m phraser.eval.sweep_postproc --dataset salami
"""
import argparse
import hashlib
import itertools
import json

import numpy as np

from phraser.eval.boundaries import pick_boundaries, snap_to_onsets, to_segments
from phraser.eval.metrics import segment_scores, beat_scores
from phraser.paths import BENCH

BASE = BENCH


def merge_same(intervals, labels):
    if len(labels) == 0:
        return intervals, labels
    out_i, out_l = [list(intervals[0])], [labels[0]]
    for iv, l in zip(intervals[1:], labels[1:]):
        if l == out_l[-1]:
            out_i[-1][1] = iv[1]
        else:
            out_i.append(list(iv))
            out_l.append(l)
    return np.array(out_i), out_l


def decode(row, thr, gap, snap, merge):
    seg_prob = np.asarray(row["seg_prob_u8"], dtype=float) / 255.0
    type_prob = np.asarray(row["type_prob_u8"], dtype=float) / 255.0
    onset = np.asarray(row["onset_act_u8"], dtype=float) / 255.0
    bounds = pick_boundaries(seg_prob, threshold=thr, min_gap_s=gap)
    if snap:
        bounds = snap_to_onsets(bounds, onset)
    est_int, est_lab = to_segments(bounds, type_prob, row["duration"])
    if merge and len(est_lab):
        est_int, est_lab = merge_same(est_int, est_lab)
    return est_int, est_lab


def score_rows(rows, thr, gap, snap, merge):
    agg = {}
    for row in rows:
        try:
            est_int, est_lab = decode(row, thr, gap, snap, merge)
            if len(est_lab) == 0:
                continue
            s = segment_scores(np.array(row["ref_intervals"]), row["ref_labels"], est_int, est_lab)
            for k, v in s.items():
                agg.setdefault(k, []).append(v)
        except Exception:
            pass
    return {k: float(np.mean(v)) for k, v in agg.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["salami", "harmonix"], required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(f"{BASE}/acts_{args.dataset}.jsonl")]
    if args.dataset == "harmonix":
        sp = json.load(open(f"{BASE}/harmonix_ft/split.json"))
        val = [r for r in rows if r["id"] in set(sp["val"])]
        test = [r for r in rows if r["id"] in set(sp["test"])]
    else:
        val = [r for r in rows if int(hashlib.md5(r["id"].encode()).hexdigest(), 16) % 4 == 0]
        test = [r for r in rows if int(hashlib.md5(r["id"].encode()).hexdigest(), 16) % 4 != 0]
    print(f"{args.dataset}: val={len(val)} test={len(test)}")

    base_cfg = (0.3, 4.0, True, False)
    base_val = score_rows(val, *base_cfg)
    print("baseline(val)", {k: round(v, 3) for k, v in base_val.items() if k in ('HR.5F', 'HR3F', 'PWF', 'Sf')})

    best_cfg, best_score = base_cfg, -1
    for thr, gap, snap, merge in itertools.product(
            [0.2, 0.25, 0.3, 0.35, 0.4, 0.5], [3.0, 5.0, 8.0, 12.0], [True, False], [True, False]):
        s = score_rows(val, thr, gap, snap, merge)
        if not s:
            continue
        obj = (s.get("HR.5F", 0) + s.get("HR3F", 0) + s.get("PWF", 0)) / 3
        if obj > best_score:
            best_score, best_cfg = obj, (thr, gap, snap, merge)
    print(f"best cfg (val): thr={best_cfg[0]} gap={best_cfg[1]} snap={best_cfg[2]} merge={best_cfg[3]} obj={best_score:.3f}")

    for name, cfg in [("untuned", base_cfg), ("tuned", best_cfg)]:
        s = score_rows(test, *cfg)
        print(f"TEST {name}: ", {k: round(v, 3) for k, v in s.items() if k in ('HR.5F', 'HR3F', 'PWF', 'PWP', 'PWR', 'Sf')})

    if args.dataset == "harmonix":
        try:
            from madmom.features.beats import DBNBeatTrackingProcessor
            dbn = DBNBeatTrackingProcessor(fps=100, min_bpm=55, max_bpm=215)
            g_old, g_new = [], []
            for row in test:
                if len(row.get("ref_beats", [])) < 10:
                    continue
                act = np.asarray(row["beat_act_u8"], dtype=float) / 255.0
                try:
                    beats = dbn(act.astype(np.float32))
                    if len(beats) > 1:
                        g_new.append(beat_scores(np.array(row["ref_beats"]), beats)["beat_F"])
                except Exception:
                    pass
            print(f"DBN beat_F (test): {np.mean(g_new):.3f} over n={len(g_new)}")
        except Exception as e:
            print("DBN unavailable:", repr(e)[:120])


if __name__ == "__main__":
    main()
