"""Dump phraser's raw activations for post-processing sweeps (from cached embeddings).

Per track: 1 Hz boundary probs (u8), 1 Hz section-type argmax+probs (u8),
100 Hz beat + onset activations (u8), plus references for scoring.
  --dataset salami   -> embeddings from salami_emb_cache; refs from SALAMI loader
  --dataset harmonix -> embeddings from harmonix_ft/emb; refs from labels jsons
Output: $PHRASER_DATA/benchmarks/acts_<dataset>.jsonl (CPU-sweepable forever).
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from phraser.paths import BENCH, CKPT as PATHS_CKPT

BASE = BENCH
CKPT = PATHS_CKPT


def u8(x):
    return np.clip(np.round(np.asarray(x) * 255), 0, 255).astype(int).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["salami", "harmonix"], required=True)
    args = ap.parse_args()

    from phraser.modules.phraser import PhraserModel
    model = PhraserModel(128, 16)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in ck["state_dict"].items()}
    model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("muq.")}, strict=False)
    model = model.cuda().eval()

    items = []
    if args.dataset == "salami":
        from phraser.eval import datasets as bench_datasets
        tracks = {t["id"]: t for t in bench_datasets.load_salami(f"{BASE}/salami-data-public")}
        for p in sorted(glob.glob(f"{BASE}/salami_emb_cache/*.npy")):
            tid = os.path.splitext(os.path.basename(p))[0]
            if tid in tracks:
                t = tracks[tid]
                items.append((tid, p, t["ref_intervals"].tolist(), t["ref_labels"], []))
    else:
        for p in sorted(glob.glob(f"{BASE}/harmonix_ft/emb/*.npy")):
            tid = os.path.splitext(os.path.basename(p))[0]
            lp = f"{BASE}/harmonix_ft/labels/{tid}.json"
            if os.path.exists(lp):
                lab = json.load(open(lp))
                ref_int = [[max(s, 0.0), e] for s, e, _ in lab["segments"]]
                ref_lab = [l for _, _, l in lab["segments"]]
                items.append((tid, p, ref_int, ref_lab, lab["beats"]))

    out = f"{BASE}/acts_{args.dataset}.jsonl"
    n = 0
    with open(out, "w") as fout, torch.no_grad():
        for tid, p, ref_int, ref_lab, ref_beats in items:
            try:
                emb = np.load(p).astype(np.float32)
                L = emb.shape[1] - (emb.shape[1] % 25)
                x = torch.from_numpy(emb[:, :L]).unsqueeze(0).cuda()
                (split_l, _sil, sect_l), _, (beat_l, onset_l) = model.forward_encoded(x)
                row = {
                    "id": tid, "duration": L / 25.0,
                    "seg_prob_u8": u8(torch.sigmoid(split_l[0, -1].squeeze(-1)).cpu().numpy()),
                    "type_prob_u8": u8(torch.softmax(sect_l[0], dim=-1).cpu().numpy()),
                    "beat_act_u8": u8(torch.sigmoid(beat_l[0].squeeze(-1)).cpu().numpy()),
                    "onset_act_u8": u8(torch.sigmoid(onset_l[0].squeeze(-1)).cpu().numpy()),
                    "ref_intervals": ref_int, "ref_labels": ref_lab, "ref_beats": ref_beats,
                }
                fout.write(json.dumps(row) + "\n")
                n += 1
                if n % 100 == 0:
                    print(n, flush=True)
            except Exception as e:
                print(f"ERR {tid}: {repr(e)[:120]}", flush=True)
    print(f"DONE {args.dataset} n={n} -> {out}")


if __name__ == "__main__":
    main()
