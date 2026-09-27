"""Preprocess Harmonix for fine-tuning: SCNet stems -> MuQ embeddings (+ labels).

Per track writes:
  $PHRASER_DATA/benchmarks/harmonix_ft/emb/<id>.npy      float16 [4, T@25Hz, 1024] (vocals, other, bass, drums)
  $PHRASER_DATA/benchmarks/harmonix_ft/labels/<id>.json  segments/beats/downbeats in the AUDIO timeline
                                                     (annotation + offset), bpm, per-stem 1 Hz RMS

Offset: the allin1-run estimate (pairwise-delta histogram on its predicted
beats — the reliable estimator; beat_F 0.787 after correction). Tracks without
an allin1 offset are skipped. Resume-safe: skips ids whose outputs already exist.
"""
import csv
import json
import os

import librosa
import numpy as np
import torch

from phraser.eval import datasets as bench_datasets
from phraser.eval.run_benchmark import build_separator, build_encoder, STEM_ORDER, MUQ_SR
from phraser.paths import BENCH, SCNET_CKPT, SCNET_CONFIG

BASE = BENCH
OUT = f"{BASE}/harmonix_ft"
os.makedirs(f"{OUT}/emb", exist_ok=True)
os.makedirs(f"{OUT}/labels", exist_ok=True)


def load_offsets(path):
    out = {}
    try:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if "offset" in r:
                    out[r["id"]] = r["offset"]
    except FileNotFoundError:
        pass
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    off_p = load_offsets(f"{BASE}/results_harmonix_full.jsonl")
    off_a = load_offsets(f"{BASE}/results_harmonix_allin1.jsonl")
    bpm = {r["File"]: float(r["BPM"]) for r in csv.DictReader(open(f"{BASE}/harmonixset/dataset/metadata.csv"))}

    tracks = bench_datasets.load_harmonix(f"{BASE}/harmonixset", audio_dir=f"{BASE}/harmonix_audio")
    with_audio = [t for t in tracks if t["audio"]]

    separate = build_separator(SCNET_CKPT,
                               SCNET_CONFIG)
    encode = build_encoder(device)

    n_ok = n_skip_off = n_done = 0
    for t in with_audio:
        tid = t["id"]
        emb_path = f"{OUT}/emb/{tid}.npy"
        lab_path = f"{OUT}/labels/{tid}.json"
        if os.path.exists(emb_path) and os.path.exists(lab_path):
            n_done += 1
            continue
        a = off_a.get(tid)
        if a is None:
            n_skip_off += 1
            continue
        offset = a
        try:
            stems = separate(t["audio"])
            embs, rms = [], {}
            for name in STEM_ORDER:
                wav = stems[name]
                e = encode(wav)  # [1, T, 1024]
                embs.append(e)
                n_sec = int(len(wav) // MUQ_SR)
                w = wav[: n_sec * MUQ_SR].reshape(n_sec, MUQ_SR)
                rms[name] = np.sqrt((w.astype(np.float64) ** 2).mean(axis=1)).round(5).tolist()
            L = min(e.shape[1] for e in embs)
            arr = torch.cat([e[:, :L] for e in embs], dim=0).numpy().astype(np.float16)  # [4, L, 1024]
            np.save(emb_path, arr)

            # beats file: time \t beat-in-bar \t bar — downbeat when beat-in-bar == 1
            beats, downbeats = [], []
            bpath = f"{BASE}/harmonixset/dataset/beats_and_downbeats/{tid}.txt"
            with open(bpath) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        tt = float(parts[0]) + offset
                        if tt >= 0:
                            beats.append(round(tt, 3))
                            if parts[1] == "1":
                                downbeats.append(round(tt, 3))
            segments = [[round(float(s) + offset, 3), round(float(e2) + offset, 3), lab]
                        for (s, e2), lab in zip(t["ref_intervals"], t["ref_labels"])]
            with open(lab_path, "w") as f:
                json.dump({"id": tid, "offset": round(offset, 3), "bpm": bpm.get(tid),
                           "duration_emb_s": L / 25.0, "segments": segments,
                           "beats": beats, "downbeats": downbeats, "stem_rms_1hz": rms}, f)
            n_ok += 1
            if n_ok % 20 == 0:
                print(f"prepped {n_ok} (skip_offset={n_skip_off}, already={n_done})", flush=True)
        except Exception as e:
            print(f"ERROR {tid}: {repr(e)[:200]}", flush=True)
    print(f"DONE prepped={n_ok} already={n_done} skip_offset={n_skip_off}", flush=True)


if __name__ == "__main__":
    main()
