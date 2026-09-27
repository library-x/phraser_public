"""Full separate->embed->segment benchmark runner.

Usage:
  python -m phraser.eval.run_benchmark --dataset harmonix --root $PHRASER_DATA/benchmarks/harmonixset \
      --audio-dir $PHRASER_DATA/benchmarks/harmonix_audio --ckpt <best.ckpt> --out results_harmonix.jsonl

Pipeline per track: audio -> SCNet 4-stem separation -> 24 kHz resample ->
MuQ embeddings in 30 s windows -> PhraserModel -> boundaries/labels/beats -> mir_eval.
Tracks without audio are skipped and counted, never silently dropped.
"""
import argparse
import json

import librosa
import numpy as np
import torch

from phraser.eval import datasets as bench_datasets
from phraser.eval.boundaries import predict_track
from phraser.eval.metrics import segment_scores, beat_scores

MUQ_SR = 24000
WINDOW_S = 30
STEM_ORDER = ["vocals", "other", "bass", "drums"]  # model input channel order


def build_separator(scnet_ckpt, scnet_config):
    """SCNet separator from the vendored package. Returns fn: path -> {stem: wav24k mono}."""
    import yaml
    from ml_collections import ConfigDict
    from phraser.scnet.SCNet import SCNet
    from phraser.scnet.inference import Seperator

    with open(scnet_config) as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
    net = SCNet(**config.model)
    net.eval()
    sep = Seperator(net, scnet_ckpt)

    def _sep(path):
        wav, sr = librosa.load(path, sr=None, mono=False)
        arr = wav.T if wav.ndim > 1 else wav[:, None]
        stems, rates = sep.separate_music_file(arr, sr)
        out = {}
        for name in STEM_ORDER:
            s = stems[name]
            if s.ndim > 1:
                s = s.mean(axis=1)
            out[name] = librosa.resample(s, orig_sr=rates[name], target_sr=MUQ_SR)
        return out
    return _sep


def build_encoder(device):
    from muq import MuQ
    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(device).eval()

    def _enc(wav):
        chunks = []
        step = WINDOW_S * MUQ_SR
        for s in range(0, len(wav), step):
            seg = wav[s:s + step]
            if len(seg) < MUQ_SR:  # <1s tail
                break
            with torch.no_grad():
                x = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).to(device)
                chunks.append(muq(x).last_hidden_state.cpu())
        return torch.cat(chunks, dim=1)  # [1, T@25Hz, 1024]
    return _enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["harmonix", "salami", "bhx"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--audio-dir", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scnet-ckpt", required=True)
    ap.add_argument("--scnet-config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="append to --out, skipping ids already present")
    ap.add_argument("--emb-cache", default=None, help="dir to save/load per-track MuQ embeddings (fp16 npy)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = {"harmonix": bench_datasets.load_harmonix, "salami": bench_datasets.load_salami,
              "bhx": bench_datasets.load_bhx}[args.dataset]
    tracks = loader(args.root, audio_dir=args.audio_dir)
    with_audio = [t for t in tracks if t["audio"]]
    print(f"{args.dataset}: {len(tracks)} annotated, {len(with_audio)} with audio")
    if args.limit:
        with_audio = with_audio[:args.limit]

    from phraser.modules.phraser import PhraserModel
    model = PhraserModel(128, 16)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("muq.")}, strict=False)
    model = model.to(device).eval()

    separate = build_separator(args.scnet_ckpt, args.scnet_config)
    encode = build_encoder(device)

    done = set()
    if args.resume:
        try:
            with open(args.out) as f:
                done = {json.loads(l).get("id") for l in f if l.strip()}
            print(f"resume: skipping {len(done)} already-scored tracks", flush=True)
        except FileNotFoundError:
            pass

    from phraser.eval.boundaries import estimate_offset
    agg, n = {}, 0
    with open(args.out, "a" if args.resume else "w") as fout:
        for t in with_audio:
            if t["id"] in done:
                continue
            try:
                cache_p = None
                if args.emb_cache:
                    import os as _os
                    _os.makedirs(args.emb_cache, exist_ok=True)
                    cache_p = f"{args.emb_cache}/{t['id']}.npy"
                if cache_p and _os.path.exists(cache_p):
                    arr = np.load(cache_p).astype(np.float32)
                    x = torch.from_numpy(arr).unsqueeze(0).to(device)
                    L = arr.shape[1]
                else:
                    stems = separate(t["audio"])
                    embs = [encode(stems[name]) for name in STEM_ORDER]
                    L = min(e.shape[1] for e in embs)
                    L -= L % 25  # keep divisible by the 25x downsample path
                    x = torch.cat([e[:, :L] for e in embs], dim=0).unsqueeze(0).to(device)  # [1,4,L,1024]
                    if cache_p:
                        np.save(cache_p, x[0].cpu().numpy().astype(np.float16))
                _mix_dir = __import__("os").environ.get("PHRASER_MIX_EMB_CACHE")
                if _mix_dir:  # 5th (aggregate) input channel = MuQ of the mixture
                    mix = np.load(f"{_mix_dir}/{t['id']}.npy").astype(np.float32)[:, :L]
                    mix = np.pad(mix, ((0, 0), (0, L - mix.shape[1]), (0, 0)))
                    x = torch.cat([x, torch.from_numpy(mix).unsqueeze(0).to(device)], dim=1)
                duration = L / 25.0
                est_int, est_lab, est_beats, beat_prob = predict_track(model, x, duration)
                # YouTube-sourced audio is offset vs. the annotation timeline; estimate a
                # constant per-track shift from the model's own beat activations.
                offset = estimate_offset(est_beats, t["ref_beats"], max_shift_s=30.0)
                if offset != 0.0:
                    est_int = np.clip(est_int - offset, 0.0, None)
                    est_beats = np.asarray(est_beats) - offset
                    est_beats = est_beats[est_beats > 0]
                row = {"id": t["id"], "offset": round(offset, 2),
                       "est_intervals": np.round(est_int, 2).tolist(),
                       "est_labels": est_lab,
                       "est_beats": np.round(np.asarray(est_beats), 2).tolist(),
                       "beat_act_u8": np.round(np.asarray(beat_prob) * 255).astype(int).tolist()}
                try:  # keep raw dumps even if scoring throws — re-scorable without GPU
                    row.update(segment_scores(t["ref_intervals"], t["ref_labels"], est_int, est_lab))
                    if len(t["ref_beats"]) > 1 and len(est_beats) > 1:
                        row.update(beat_scores(t["ref_beats"], est_beats))
                except Exception as e:
                    row["error"] = repr(e)[:300]
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                n += 1
                for k, v in row.items():
                    if isinstance(v, (int, float)):
                        agg.setdefault(k, []).append(v)
                if n % 10 == 0:
                    print(n, {k: round(float(np.mean(v)), 3) for k, v in agg.items()}, flush=True)
            except Exception as e:
                fout.write(json.dumps({"id": t["id"], "error": repr(e)[:300]}) + "\n")
    print("FINAL", n, {k: round(float(np.mean(v)), 3) for k, v in agg.items()})


if __name__ == "__main__":
    main()
