"""Score the retrained All-In-One checkpoint on the shared Harmonix test split
through OUR harness (same refs as eval_from_emb: audio-timeline labels jsons).

Run in the allin1 env:
  python -m phraser.eval.eval_allin1_retrained --ckpt <lightning.ckpt>
"""
import argparse
import json

import numpy as np
import torch
from omegaconf import OmegaConf

from phraser.eval.datasets import map_label
from phraser.eval.metrics import segment_scores, beat_scores
from phraser.paths import BENCH

BASE = BENCH
FT = f"{BASE}/harmonix_ft"
TRAIN_DIR = f"{BASE}/allin1_train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=f"{FT}/eval_allin1_retrained.jsonl")
    args = ap.parse_args()

    from allin1.models.allinone import AllInOne
    from allin1.helpers import run_inference
    from pathlib import Path

    ck = torch.load(args.ckpt, map_location="cuda")
    if "hyper_parameters" in ck and ck["hyper_parameters"]:
        cfg = OmegaConf.create(ck["hyper_parameters"]["cfg"])
    else:  # trainer did not save hparams; training used dataclass defaults + path overrides
        from allin1.config import Config, HarmonixConfig
        cfg = OmegaConf.structured(Config(data=HarmonixConfig()))
    sd = ck["state_dict"]
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    model = AllInOne(cfg).to("cuda")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing:", [m for m in missing][:5], "unexpected:", [u for u in unexpected][:5])
    model.eval()

    split = json.load(open(f"{FT}/split.json"))
    agg, n = {}, 0
    with open(args.out, "w") as fout, torch.no_grad():
        for tid in split["test"]:
            spec = Path(f"{TRAIN_DIR}/features/{tid}.npy")
            if not spec.exists():
                continue
            try:
                lab = json.load(open(f"{FT}/labels/{tid}.json"))
                res = run_inference(path=Path(f"{TRAIN_DIR}/tracks/{tid}.mp3"), spec_path=spec,
                                    model=model, device="cuda",
                                    include_activations=False, include_embeddings=False)
                est_int = np.array([[s.start, s.end] for s in res.segments])
                est_lab = [map_label(s.label) for s in res.segments]
                est_beats = np.asarray(res.beats, dtype=float)
                ref_int = np.clip(np.array([[s, e] for s, e, _ in lab["segments"]], dtype=float), 0, None)
                ref_lab = [map_label(l) for _, _, l in lab["segments"]]
                row = {"id": tid, **segment_scores(ref_int, ref_lab, est_int, est_lab)}
                if len(lab["beats"]) > 1 and len(est_beats) > 1:
                    row.update(beat_scores(np.array(lab["beats"]), est_beats))
                fout.write(json.dumps(row) + "\n")
                n += 1
                for k, v in row.items():
                    if isinstance(v, (int, float)):
                        agg.setdefault(k, []).append(v)
            except Exception as e:
                fout.write(json.dumps({"id": tid, "error": repr(e)[:200]}) + "\n")
    print(f"EVAL retrained-allin1 n={n}")
    print("RESULT", {k: round(float(np.mean(v)), 3) for k, v in agg.items()})


if __name__ == "__main__":
    main()
