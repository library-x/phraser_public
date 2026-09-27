"""Idempotently patch allin1's HarmonixDataset to honour ALLIN1_SPLIT_JSON.

When the env var points at a split.json ({"train": [...], "val": [...],
"test": [...]}), track selection uses it (intersected with available tracks);
otherwise the original modulo-fold logic applies. Patch is marked and applied
once; the original file is backed up alongside.
"""
import importlib.util
import os
import re
import shutil

PATH = os.path.join(os.path.dirname(importlib.util.find_spec("allin1").origin),
                    "training", "data", "datasets", "harmonix", "dataset.py")
MARKER = "ALLIN1_SPLIT_JSON patch"

NEW_BLOCK = '''    import os as _os, json as _json  # ALLIN1_SPLIT_JSON patch
    track_ids = sorted(t.stem for t in Path(cfg.data.path_track_dir).glob('*.mp3'))
    _sj = _os.environ.get('ALLIN1_SPLIT_JSON')
    if _sj:
      _want = set(_json.load(open(_sj))[split])
      track_ids = [tid for tid in track_ids if tid in _want]
    else:
      fold = cfg.fold
      folds = np.arange(len(track_ids)) % cfg.total_folds
      test_fold = fold
      val_fold = (fold + 1) % cfg.total_folds
      if split == 'train':
        track_ids = [tid for tid, f2 in zip(track_ids, folds) if f2 not in [test_fold, val_fold]]
      elif split == 'val':
        track_ids = [tid for tid, f2 in zip(track_ids, folds) if f2 == val_fold]
      elif split == 'test':
        track_ids = [tid for tid, f2 in zip(track_ids, folds) if f2 == test_fold]
      else:
        raise ValueError('Unknown dataset split: ' + split)'''


def main():
    src = open(PATH).read()
    if MARKER in src:
        print("already patched")
        return
    pat = re.compile(
        r"    fold = cfg\.fold\n.*?raise ValueError\(f'Unknown dataset split: \{split\}'\)",
        re.DOTALL)
    if not pat.search(src):
        raise SystemExit("PATTERN NOT FOUND — dataset.py layout changed, patch manually")
    shutil.copy(PATH, PATH + ".orig")
    open(PATH, "w").write(pat.sub(NEW_BLOCK.replace("\\", "\\\\"), src, count=1))
    print("patched OK (backup at dataset.py.orig)")


if __name__ == "__main__":
    main()
