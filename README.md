# PHRASER: Loop-Precise Stem-Aware Music Structure Segmentation

Code, weights and data for the ICASSP 2027 submission
*PHRASER: Loop-Precise Stem-Aware Music Structure Segmentation*
(Daniel Dobrowolski, Maciej Kurek, Filip Sikora — Tuney).

PHRASER takes separated stems as input **and** predicts per-stem structure. A frozen
MuQ encoder feeds a 2D neighborhood-attention U-Net over a (stem × time) grid, with a
self-similarity transformer at the coarse bottleneck. Heads predict section boundaries
and labels, per-stem entry/exit and silence, beats and onsets. Coarse boundaries are
snapped to onset peaks, so cuts land within one 5 ms step of the metrical grid.

## What is released

| | |
|---|---|
| Code | this repository (model, training loop, full evaluation harness, Tuney-Small pipeline) |
| Weights | [TuneyCompany/phraser](https://huggingface.co/TuneyCompany/phraser): PHRASER trained on Tuney-Small (31M params) |
| Data | [TuneyCompany/tuney-small](https://huggingface.co/datasets/TuneyCompany/tuney-small): 504 grid-exact tracks, 22 genres, 426 train / 78 held-out |

The full 9k-track Tuney corpus and the model trained on it are proprietary and not
released. Tuney-Small reproduces the matched-training comparison (the Tuney-Small rows
of Tables 1 and 2 in the paper).

## Install

    uv venv --python 3.12
    uv pip install -e .                       # inference deps
    uv pip install -e .[train,eval]           # + training/eval extras
    # NATTEN: pick the wheel matching your torch/CUDA build (after torch):
    uv pip install natten -f https://shi-labs.com/natten/wheels
    # optional, DBN beat decoding:
    uv pip install cython && uv pip install "git+https://github.com/CPJKU/madmom"

Separation uses SCNet-large; put its weights at `checkpoints/SCNet-large.th`
(or set `PHRASER_SCNET`). MuQ (`OpenMuQ/MuQ-large-msd-iter`) downloads from the Hugging Face Hub.

## Download weights and data

The dataset is gated: accept its terms on the
[dataset page](https://huggingface.co/datasets/TuneyCompany/tuney-small) and run `hf auth login` first.

    from huggingface_hub import hf_hub_download, snapshot_download
    hf_hub_download("TuneyCompany/phraser", "phraser_tuney_small.ckpt", local_dir="weights")
    snapshot_download("TuneyCompany/tuney-small", repo_type="dataset", local_dir="tuney_small")

These land in `./weights` and `./tuney_small`, the default paths below.

## Paths

Nothing is hard-coded; see [phraser/paths.py](phraser/paths.py).

| variable | default | meaning |
|---|---|---|
| `PHRASER_DATA` | `./data` | benchmarks, caches, outputs |
| `PHRASER_TSMALL` | `./tuney_small` | Tuney-Small root |
| `PHRASER_CKPT` | `./weights/phraser_tuney_small.ckpt` | PHRASER weights |
| `PHRASER_SCNET` | `./checkpoints/SCNet-large.th` | separator weights |
| `PHRASER_CORPUS` | `$PHRASER_DATA/tuney` | any corpus in Tuney format, for training |
| `PHRASER_RUNS` | `./runs` | TensorBoard runs |

## Reproduce the Tuney-Small result

    python -m phraser.tuney_small.separate_tuney_small     # SCNet stems -> tuney_small_<genre>/separated/
    python -m phraser.tuney_small.muq_tuney_small          # MuQ embeddings -> tuney_small_<genre>/encoded_short/
    python -m phraser.tuney_small.eval_ts phraser weights/phraser_tuney_small.ckpt

`eval_ts` scores boundary F at ±0.5 s / ±3 s plus PWF / Sf on the 78 held-out tracks, and
writes per-track estimates to `tuney_small/est_phraser.jsonl`. `eval_ts score <name> <est.jsonl>`
scores any other system's estimates with the same harness.

To retrain on Tuney-Small (80 epochs, one A100, about 2 h):

    PHRASER_DATA_GLOB="$PHRASER_TSMALL/tuney_small_*/splits.json" \
        python -m phraser.train --exp_name tuney_small_phraser --epochs 80 --warmup 8 --seed 42 --batch_size 6 --save_freq 5

Baselines on Tuney-Small, retrained with their own code on the same split:
`a1_prep_ts.py` / `gen_a1_beats.py` / `a1_infer_ts.py` (All-In-One), `sf_prep_ts.py` /
`score_sf_a1.py` (SongFormer), and `gemini_tsmall.py` (Gemini; needs `GEMINI_API_KEY`).
`verify_disjoint.py` checks that no loop element in the held-out split occurs in training.

## Tuney-Small format

    tuney_small/
      manifest.json        # "<genre>/<id>" -> genre, bpm, key, n_segments, paths
      splits.json          # global split: train / val (val is element-disjoint from train)
      refs_val.json        # reference segments for the held-out split (seconds)
    tuney_small_<genre>/
      splits.json          # per-genre train / test (test = the global val)
      parsed/<id>.mp3      # the mix
      parsed/<id>.json     # exact labels

Each annotation holds `bpm`, `key`, `segmnets` (sic: section `start`/`end` in **bars**,
and `name`), `stem_elements` (per stem, which loop elements play from bar `start` to `end`),
and `elements` (`[element_id, register]`). Multiply bars by `240 / bpm` to get seconds.
Every boundary, stem entry/exit, beat and bar is exact, because the tracks are built from loops.

## Evaluation harness (`phraser/eval/`)

Public benchmarks (Harmonix / SongFormBench-HX, SALAMI) and the controlled retraining studies:

- `run_benchmark.py`: audio → SCNet → MuQ → model → scores, with raw-prediction dumps
- `baseline_allin1.py`, `baseline_songformer.py`: released baselines on the same audio and references
- `datasets.py`, `metrics.py`, `boundaries.py`: annotation adapters, mir_eval wrappers, decoding
- `prep_harmonix_ft.py`, `finetune_harmonix.py`, `eval_from_emb.py`: PHRASER trained on Harmonix
- `setup_allin1_train.py`, `patch_allin1_split.py`, `eval_allin1_retrained.py`: All-In-One retrained with its own code
- `score_songformer_retrained.py`, `score_bhx.py`: SongFormer retrained, SongFormBench-HX scoring
- `loopability_eval.py`, `stem_loopability_eval.py`: loopability and per-stem loop recovery
- `sweep_postproc.py`, `dump_acts.py`: validation-only decoding sweeps

## Tests

    uv pip install -e .[eval,dev]
    pytest tests/

These are CPU-only unit tests for the harness (metrics, decoding, offset estimation, annotation parsers).

## Licenses

- Code: MIT, see [LICENSE](LICENSE). Vendored third-party code keeps its own license, see
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- Weights: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). They build on MuQ,
  whose weights are also non-commercial.
- Tuney-Small: [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). You may use it
  for non-commercial research, including training and evaluation. You may not share modified
  audio, such as separated stems or excerpts. The mixes contain third-party library samples:
  do not extract or redistribute them.

## Citation

    @inproceedings{phraser2027,
      title     = {{PHRASER}: Loop-Precise Stem-Aware Music Structure Segmentation},
      author    = {Dobrowolski, Daniel and Kurek, Maciej and Sikora, Filip},
      booktitle = {Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      year      = {2027},
      note      = {Under review}
    }
