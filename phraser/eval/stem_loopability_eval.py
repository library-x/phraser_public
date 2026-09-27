"""Stem-level loop recovery on the Tuney test set (GPU pass + inline scoring).

Ground truth per stem: the stem_elements each track was constructed from
(start/end in bars -> seconds). For the supervised stems (other, bass, drums):

  element recall@tol  fraction of true stem elements with BOTH endpoints
                      matched by a predicted stem boundary within tol
  boundary F@tol      set F-measure between predicted stem boundaries and
                      true stem element edges

Systems:
  phraser  per-stem entry/exit heads decoded per stem (threshold + min-gap +
           onset snap), i.e. genuine stem-level segmentation
  allin1   its mixture-level boundaries replicated to every stem (the best a
           track-level system can structurally offer)

Output: aggregate table + per-track jsonl at $PHRASER_DATA/benchmarks/stem_loop.jsonl
"""
import glob
import json
import os

import numpy as np
import torch

from phraser.eval.boundaries import pick_boundaries, snap_to_onsets
from phraser.paths import BENCH, CKPT as PATHS_CKPT, TUNEY_ALLIN1, corpus_splits

CKPT = PATHS_CKPT
ALLIN1_DUMPS = f"{TUNEY_ALLIN1}/eval_allin1_tuney_dumps.jsonl"
OUT = f"{BENCH}/stem_loop.jsonl"
TOL = 0.005
STEMS = {"other": 1, "bass": 2, "drums": 3}  # prediction channel indices (supervised)


def match(pred, truth, tol=TOL):
    pred = np.sort(np.asarray(pred, dtype=float))
    hits = np.zeros(len(truth), dtype=bool)
    if len(pred) == 0:
        return hits
    for i, t in enumerate(truth):
        j = np.searchsorted(pred, t)
        for k in (j - 1, j):
            if 0 <= k < len(pred) and abs(pred[k] - t) <= tol:
                hits[i] = True
    return hits


def stem_truth(d, bar_len):
    out = {}
    for name in STEMS:
        els = d["stem_elements"].get(name, [])
        out[name] = [(float(r["start"]) * bar_len, float(r["end"]) * bar_len) for r in els]
    return out


def score_stem(bounds, elements):
    if not elements:
        return None
    edges = sorted({round(e, 4) for el in elements for e in el})
    hit_edges = match(bounds, edges)
    el_rec = float(np.mean([match(bounds, [a]).all() and match(bounds, [b]).all()
                            for a, b in elements]))
    prec_hits = match(edges, bounds)
    p = float(prec_hits.mean()) if len(bounds) else 0.0
    r = float(hit_edges.mean())
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"el_recall": el_rec, "bound_f": f, "n_el": len(elements)}


def main():
    from phraser.modules.phraser import PhraserModel
    model = PhraserModel(128, 16)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in ck["state_dict"].items()}
    model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("muq.")}, strict=False)
    model = model.cuda().eval()

    a1 = {}
    for line in open(ALLIN1_DUMPS):
        r = json.loads(line)
        if "est_intervals" in r:
            edges = sorted({round(float(x), 3) for iv in r["est_intervals"] for x in iv})
            a1[r["id"]] = edges

    agg = {}
    n = 0
    with open(OUT, "w") as fout, torch.no_grad():
        for ds in corpus_splits():
            base = os.path.dirname(ds)
            for tid in json.load(open(ds))["test"]:
                try:
                    d = json.load(open(f"{base}/parsed/{tid}.json"))
                    bar_len = 240.0 / float(d["bpm"])
                    truth = stem_truth(d, bar_len)
                    embs = [np.load(f"{base}/encoded_short/{tid}_{s}.npy")
                            for s in ("vocals", "other", "bass", "drums")]
                    L = min(e.shape[1] for e in embs)
                    L -= L % 25
                    x = torch.from_numpy(np.concatenate([e[:, :L] for e in embs], axis=0)
                                         .astype(np.float32)).unsqueeze(0).cuda()
                    (split_l, _sil, _sect), _, (_beat, onset_l) = model.forward_encoded(x)
                    onset = torch.sigmoid(onset_l[0].squeeze(-1)).cpu().numpy()
                    row = {"id": tid}
                    for name, ch in STEMS.items():
                        prob = torch.sigmoid(split_l[0, ch].squeeze(-1)).cpu().numpy()
                        b = snap_to_onsets(pick_boundaries(prob, threshold=0.3, min_gap_s=2.0), onset)
                        s = score_stem(b, truth[name])
                        if s:
                            row[f"ph_{name}"] = s
                        sa = score_stem(a1.get(tid, []), truth[name])
                        if sa:
                            row[f"a1_{name}"] = sa
                    fout.write(json.dumps(row) + "\n")
                    n += 1
                    for k, v in row.items():
                        if isinstance(v, dict):
                            for kk, vv in v.items():
                                if kk != "n_el":
                                    agg.setdefault(f"{k}.{kk}", []).append(vv)
                    if n % 200 == 0:
                        print(n, flush=True)
                except Exception as e:
                    fout.write(json.dumps({"id": tid, "error": repr(e)[:120]}) + "\n")
    print(f"DONE n={n}")
    for k in sorted(agg):
        print(f"{k}: {np.mean(agg[k]):.3f}")


if __name__ == "__main__":
    main()
