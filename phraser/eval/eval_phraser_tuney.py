"""Phraser HR on the Tuney test set (in-domain), from existing encoded_short
embeddings — mir_eval boundary detection, same convention as the Harmonix study.

GT boundaries = every segment-block end in seconds (bpm * bars), matching the
supervision convention of BOTH models in this comparison.
Run in phraser env: python -u -m phraser.eval.eval_phraser_tuney
"""
import glob
import json
import os

import numpy as np
import torch

from phraser.eval.boundaries import predict_track
from phraser.eval.metrics import segment_scores
from phraser.paths import BENCH, CKPT as PATHS_CKPT, corpus_splits

CKPT = PATHS_CKPT
OUT = f"{BENCH}/results_tuney_phraser.jsonl"
LBLMAP = {"intro": "intro", "outro": "outro", "verse": "verse", "chorus": "chorus",
          "bridge": "bridge", "silence": "silence"}


def main():
    from phraser.modules.phraser import PhraserModel
    model = PhraserModel(128, 16)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in ck["state_dict"].items()}
    model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("muq.")}, strict=False)
    model = model.cuda().eval()

    agg, n = {}, 0
    with open(OUT, "w") as fout:
        for ds in corpus_splits():
            base = os.path.dirname(ds)
            for tid in json.load(open(ds))["test"]:
                try:
                    jp = f"{base}/parsed/{tid}.json"
                    embs = []
                    for s in ("vocals", "other", "bass", "drums"):
                        embs.append(np.load(f"{base}/encoded_short/{tid}_{s}.npy"))
                    L = min(e.shape[1] for e in embs)
                    L -= L % 25
                    x = torch.from_numpy(np.concatenate([e[:, :L] for e in embs], axis=0)
                                         .astype(np.float32)).unsqueeze(0).cuda()
                    duration = L / 25.0
                    est_int, est_lab, _beats, _bp = predict_track(model, x, duration)

                    d = json.load(open(jp))
                    bar_len = 240.0 / float(d["bpm"])
                    segs = sorted(((float(r["start"]) * bar_len, float(r["end"]) * bar_len,
                                    LBLMAP.get(str(r["name"]).lower().rstrip("0123456789"), "verse"))
                                   for r in d["segmnets"]), key=lambda z: z[0])
                    ref_int = np.array([[s, e] for s, e, _ in segs])
                    ref_lab = [l for _, _, l in segs]
                    row = {"id": tid, **segment_scores(ref_int, ref_lab, est_int, est_lab)}
                    fout.write(json.dumps(row) + "\n")
                    n += 1
                    for k, v in row.items():
                        if isinstance(v, (int, float)):
                            agg.setdefault(k, []).append(v)
                    if n % 100 == 0:
                        print(n, {k: round(float(np.mean(v)), 3) for k, v in agg.items()}, flush=True)
                except Exception as e:
                    fout.write(json.dumps({"id": tid, "error": repr(e)[:150]}) + "\n")
    print(f"FINAL n={n}", {k: round(float(np.mean(v)), 3) for k, v in agg.items()})


if __name__ == "__main__":
    main()
