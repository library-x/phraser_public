"""Evaluate a phraser checkpoint on the Harmonix fine-tune test split, straight
from precomputed embeddings (no audio pipeline). Refs come from the offset-
corrected labels jsons, so everything lives in the audio timeline.

  python -u -m phraser.eval.eval_from_emb --ckpt <path> [--split test]
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from phraser.eval.boundaries import predict_track
from phraser.eval.metrics import segment_scores, beat_scores
from phraser.paths import BENCH

BASE = os.path.join(BENCH, "harmonix_ft")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from phraser.modules.phraser import PhraserModel
    model = PhraserModel(128, 16)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("muq.")}, strict=False)
    model = model.cuda().eval()

    ids = json.load(open(f"{BASE}/split.json"))[args.split]
    agg, n = {}, 0
    rows = []
    for tid in ids:
        try:
            lab = json.load(open(f"{BASE}/labels/{tid}.json"))
            emb = np.load(f"{BASE}/emb/{tid}.npy").astype(np.float32)
            L = emb.shape[1] - (emb.shape[1] % 25)
            x = torch.from_numpy(emb[:, :L]).unsqueeze(0).cuda()
            duration = L / 25.0
            est_int, est_lab, est_beats, _ = predict_track(model, x, duration)
            ref_int = np.array([[s, e] for s, e, _ in lab["segments"]], dtype=float)
            ref_int = np.clip(ref_int, 0.0, None)
            ref_lab = [l for _, _, l in lab["segments"]]
            row = {"id": tid, **segment_scores(ref_int, ref_lab, est_int, est_lab)}
            if len(lab["beats"]) > 1 and len(est_beats) > 1:
                row.update(beat_scores(np.array(lab["beats"]), est_beats))
            rows.append(row)
            n += 1
            for k, v in row.items():
                if isinstance(v, (int, float)):
                    agg.setdefault(k, []).append(v)
        except Exception as e:
            rows.append({"id": tid, "error": repr(e)[:200]})
    print(f"EVAL {os.path.basename(args.ckpt)} split={args.split} n={n}")
    print("RESULT", {k: round(float(np.mean(v)), 3) for k, v in agg.items()})
    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
