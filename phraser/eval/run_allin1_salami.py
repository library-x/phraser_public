"""All-In-One on SALAMI with a retrained checkpoint (protocol-consistency run:
every system zero-shot on SALAMI after training on the Tuney corpus).

--which released: allin1.analyze on the 374-track audio subset with byproducts
kept (builds the demix/spec cache; released results are a free byproduct).
--which ckpt --ckpt <path>: run_inference with the retrained lightning ckpt
against the cached specs. No offset correction (SALAMI refs carry no beats).

Run inside the allin1 env.
"""
import argparse
import glob
import json
import os
from pathlib import Path

from phraser.eval import datasets as bench_datasets
from phraser.eval.datasets import map_label
from phraser.paths import BENCH, SONGFORMER

CACHE = f"{SONGFORMER}/salami_allin1_cache"


def iter_tracks():
    return [t for t in bench_datasets.load_salami(
        f"{BENCH}/salami-data-public",
        audio_dir=f"{BENCH}/salami_audio") if t["audio"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["released", "ckpt"], required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tracks = iter_tracks()
    print("salami tracks:", len(tracks), flush=True)
    os.makedirs(CACHE, exist_ok=True)
    rows = []

    if args.which == "released":
        import allin1
        for i, t in enumerate(tracks):
            try:
                res = allin1.analyze(t["audio"], device="cuda", include_activations=False,
                                     demix_dir=CACHE + "/demix", spec_dir=CACHE + "/spec",
                                     keep_byproducts=True)
                rows.append({"id": t["id"],
                             "est_intervals": [[float(s.start), float(s.end)] for s in res.segments],
                             "est_labels": [map_label(s.label) for s in res.segments]})
            except Exception as e:
                rows.append({"id": t["id"], "error": repr(e)[:300]})
            if (i + 1) % 25 == 0:
                print(i + 1, flush=True)
    else:
        import torch
        from omegaconf import OmegaConf
        from allin1.models.allinone import AllInOne
        from allin1.helpers import run_inference
        ck = torch.load(args.ckpt, map_location="cuda")
        if "hyper_parameters" in ck and ck["hyper_parameters"]:
            cfg = OmegaConf.create(ck["hyper_parameters"]["cfg"])
        else:
            from allin1.config import Config, HarmonixConfig
            cfg = OmegaConf.structured(Config(data=HarmonixConfig()))
        sd = ck["state_dict"]
        sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
        model = AllInOne(cfg).to("cuda")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("missing:", missing[:3], "unexpected:", unexpected[:3], flush=True)
        model.eval()
        with torch.no_grad():
            for i, t in enumerate(tracks):
                stem = Path(t["audio"]).stem
                specs = glob.glob(CACHE + "/spec/" + stem + ".npy")
                if not specs:
                    rows.append({"id": t["id"], "error": "no cached spec"})
                    continue
                try:
                    res = run_inference(path=Path(t["audio"]), spec_path=Path(specs[0]),
                                        model=model, device="cuda",
                                        include_activations=False, include_embeddings=False)
                    rows.append({"id": t["id"],
                                 "est_intervals": [[float(s.start), float(s.end)] for s in res.segments],
                                 "est_labels": [map_label(s.label) for s in res.segments]})
                except Exception as e:
                    rows.append({"id": t["id"], "error": repr(e)[:300]})
                if (i + 1) % 25 == 0:
                    print(i + 1, flush=True)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("DONE", sum(1 for r in rows if "error" not in r), "ok /", len(rows), flush=True)


if __name__ == "__main__":
    main()
