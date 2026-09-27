"""Filesystem locations, configurable via environment variables.

    PHRASER_DATA      root for benchmark data, caches and outputs  (default: ./data)
    PHRASER_CORPUS    Tuney-format corpus root: <CORPUS>/<dataset>/{splits.json,parsed/,separated/,encoded_short/}
                                                                   (default: $PHRASER_DATA/tuney)
    PHRASER_TSMALL    Tuney-Small release root (tuney_small/, tuney_small_<genre>/)
                                                                   (default: ./tuney_small)
    PHRASER_SONGFORMER_SRC  SongFormer checkout, for the retraining scripts
    PHRASER_CKPT      PHRASER checkpoint                            (default: ./weights/phraser_tuney_small.ckpt)
    PHRASER_SCNET     SCNet-large separator weights                 (default: ./checkpoints/SCNet-large.th)
    PHRASER_RUNS      TensorBoard / training run directory         (default: ./runs)
    PHRASER_HF_CACHE  Hugging Face cache for baselines              (default: HF default)
    PHRASER_EXCLUDE   comma-separated substrings; corpus datasets whose path
                      contains one are skipped                      (default: none)
"""
import glob
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

DATA = os.environ.get("PHRASER_DATA", "data")
BENCH = os.path.join(DATA, "benchmarks")
CORPUS = os.environ.get("PHRASER_CORPUS", os.path.join(DATA, "tuney"))
CKPT = os.environ.get("PHRASER_CKPT", os.path.join("weights", "phraser_tuney_small.ckpt"))
SCNET_CKPT = os.environ.get("PHRASER_SCNET", os.path.join("checkpoints", "SCNet-large.th"))
SCNET_CONFIG = os.path.join(_HERE, "scnet", "config.yaml")
RUNS = os.environ.get("PHRASER_RUNS", "runs")
HF_CACHE = os.environ.get("PHRASER_HF_CACHE") or None
TUNEY_ALLIN1 = os.path.join(DATA, "tuney_allin1")
SONGFORMER = os.path.join(DATA, "songformer_train")
SONGFORMER_SRC = os.environ.get("PHRASER_SONGFORMER_SRC", "SongFormer")
TSMALL = os.environ.get("PHRASER_TSMALL", "tuney_small")
CURVES = os.path.join(DATA, "curves")

EXCLUDE = [s for s in os.environ.get("PHRASER_EXCLUDE", "").split(",") if s]


def is_excluded(path):
    return any(s in path for s in EXCLUDE)


def corpus_splits():
    """Sorted <CORPUS>/*/splits.json, minus excluded datasets."""
    return [p for p in sorted(glob.glob(os.path.join(CORPUS, "*", "splits.json")))
            if not is_excluded(p)]
