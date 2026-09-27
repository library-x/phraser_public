"""Dump phraser's predicted segment intervals on the Tuney test set (GPU, quick).
Mirrors eval_phraser_tuney but writes est_intervals/est_beats instead of scores.
"""
import glob
import json
import os

import numpy as np
import torch

from phraser.eval.boundaries import predict_track
from phraser.paths import BENCH, CKPT as PATHS_CKPT, corpus_splits

CKPT = PATHS_CKPT
OUT = f"{BENCH}/results_tuney_phraser_est.jsonl"


def main():
    from phraser.modules.phraser import PhraserModel
    model = PhraserModel(128, 16)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in ck["state_dict"].items()}
    model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("muq.")}, strict=False)
    model = model.cuda().eval()

    n = 0
    with open(OUT, "w") as fout:
        for ds in corpus_splits():
            base = os.path.dirname(ds)
            for tid in json.load(open(ds))["test"]:
                try:
                    embs = [np.load(f"{base}/encoded_short/{tid}_{s}.npy")
                            for s in ("vocals", "other", "bass", "drums")]
                    L = min(e.shape[1] for e in embs)
                    L -= L % 25
                    x = torch.from_numpy(np.concatenate([e[:, :L] for e in embs], axis=0)
                                         .astype(np.float32)).unsqueeze(0).cuda()
                    est_int, est_lab, est_beats, _ = predict_track(model, x, L / 25.0)
                    fout.write(json.dumps({
                        "id": tid,
                        "est_intervals": np.round(est_int, 3).tolist(),
                        "est_labels": est_lab,
                        "est_beats": np.round(np.asarray(est_beats), 3).tolist(),
                    }) + "\n")
                    n += 1
                    if n % 200 == 0:
                        print(n, flush=True)
                except Exception as e:
                    fout.write(json.dumps({"id": tid, "error": repr(e)[:120]}) + "\n")
    print(f"DONE n={n}")


if __name__ == "__main__":
    main()
