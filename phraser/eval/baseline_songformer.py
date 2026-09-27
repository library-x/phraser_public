"""SongFormer (ASLP-lab, arXiv:2510.02797) as an additional baseline, scored by
our harness on our audio.

  --dataset salami    audio files, refs from SALAMI loader, no offset
  --dataset harmonix  audio files, refs from Harmonix loader, SHARED per-track
                      alignment from harmonix_alignment_fits.json (same fits
                      applied to all systems)
  --dataset tuney     mixes summed in memory from the four separated stems,
                      refs from the parsed corpus annotations

Run in the `songformer` env:
  python -u -m phraser.eval.baseline_songformer --dataset salami --out ...
"""
import argparse
import glob
import json
import os

import numpy as np

from phraser.eval import datasets as bench_datasets
from phraser.eval.datasets import map_label
from phraser.eval.metrics import segment_scores
from phraser.paths import BENCH, HF_CACHE, corpus_splits

BASE = BENCH
SR = 24000


def load_model():
    import sys
    from huggingface_hub import snapshot_download
    from transformers import AutoModel
    local_dir = snapshot_download(repo_id="ASLP-lab/SongFormer", repo_type="model",
                                  cache_dir=HF_CACHE,
                                  ignore_patterns=["SongFormer.pt"])
    sys.path.append(local_dir)
    os.environ["SONGFORMER_LOCAL_DIR"] = local_dir
    m = AutoModel.from_pretrained(local_dir, trust_remote_code=True, low_cpu_mem_usage=False)
    return m.to("cuda:0").eval()


def tuney_mix(base, tid):
    import librosa
    stems = []
    for s in ("bass", "drums", "other", "vocals"):
        w, _ = librosa.load(f"{base}/separated/{tid}_{s}.wav", sr=SR, mono=True)
        stems.append(w)
    L = max(len(w) for w in stems)
    mix = np.zeros(L, dtype=np.float32)
    for w in stems:
        mix[:len(w)] += w
    peak = np.abs(mix).max()
    if peak > 1.0:
        mix /= peak
    return mix


def iter_tracks(dataset):
    if dataset == "salami":
        for t in bench_datasets.load_salami(f"{BASE}/salami-data-public",
                                            audio_dir=f"{BASE}/salami_audio"):
            if t["audio"]:
                yield t["id"], t["audio"], np.asarray(t["ref_intervals"], float), t["ref_labels"], None
    elif dataset == "harmonix":
        fits = json.load(open(f"{BASE}/harmonix_alignment_fits.json"))
        for t in bench_datasets.load_harmonix(f"{BASE}/harmonixset",
                                              audio_dir=f"{BASE}/harmonix_audio"):
            if t["audio"] and t["id"] in fits:
                f = fits[t["id"]]
                yield t["id"], t["audio"], np.asarray(t["ref_intervals"], float), t["ref_labels"], (f["rate"], f["shift"])
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
                ref = np.array([[s, e] for s, e, _ in segs])
                labs = [l for _, _, l in segs]
                yield tid, ("TUNEYMIX", b), ref, labs, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["salami", "harmonix", "tuney"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    model = load_model()
    agg, n = {}, 0
    with open(args.out, "w") as fout:
        for tid, audio, ref_int, ref_lab, fit in iter_tracks(args.dataset):
            if args.limit and n >= args.limit:
                break
            try:
                if isinstance(audio, tuple):
                    result = model(tuney_mix(audio[1], tid))
                else:
                    result = model(str(audio))
                import ast
                segs = result if isinstance(result, list) else ast.literal_eval(str(result))
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
                fout.flush()
                n += 1
                for k, v in row.items():
                    if isinstance(v, (int, float)):
                        agg.setdefault(k, []).append(v)
                if n % 25 == 0:
                    print(n, {k: round(float(np.mean(v)), 3) for k, v in agg.items()
                              if k in ("HR.5F", "HR3F", "PWF", "Sf")}, flush=True)
            except Exception as e:
                fout.write(json.dumps({"id": tid, "error": repr(e)[:200]}) + "\n")
    print("FINAL", n, {k: round(float(np.mean(v)), 3) for k, v in agg.items()})


if __name__ == "__main__":
    main()
