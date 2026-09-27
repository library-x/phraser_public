"""Present the Tuney corpus in All-In-One's expected Harmonix-style layout.

$PHRASER_DATA/tuney_allin1/
  tracks/<id>.mp3          symlink to files/<id>.mp3
  demix/htdemucs/<id>/{bass,drums,other,vocals}.wav   symlinks to separated stems
  beats/<id>.txt           2-col (time \t beat-in-bar), grid from bpm + robust bar offset
  segments/<id>.txt        (time \t label) rows + trailing 'end', their 10-label vocab
  metadata.csv, split.json {train,val,test}

Beat/bar grids replicate src.utils.data_loader's logic exactly (find_robust_onsets
on element ends -> bar/beat phase). Label head will be trained too (our labels map
cleanly into their vocab); boundary convention = every segment-block end, same as
phraser's supervision.
"""
import csv
import glob
import json
import os

import numpy as np

from phraser.utils.data_loader import find_robust_onsets
from phraser.paths import TUNEY_ALLIN1, corpus_splits

OUT = TUNEY_ALLIN1
VOCAB_FIX = {"silence": None}  # leading silence dropped; mid silence -> break below


def norm_label(raw):
    l = str(raw).strip().lower().rstrip("0123456789")
    if l in ("intro", "outro", "verse", "chorus", "bridge"):
        return l
    if l == "silence":
        return "silence"
    return "inst"


def main():
    for sub in ("tracks", "beats", "segments", "demix/htdemucs", "features"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)

    split = {"train": [], "val": [], "test": []}
    meta_rows = []
    kept = skipped = 0
    for ds in corpus_splits():
        base = os.path.dirname(ds)
        sp = json.load(open(ds))
        for part in ("train", "test"):
            for tid in sp[part]:
                jp = f"{base}/parsed/{tid}.json"
                stems = {s: f"{base}/separated/{tid}_{s}.wav" for s in ("bass", "drums", "other", "vocals")}
                mp3s = glob.glob(f"{base}/files/{tid}*.mp3")
                # many tracks have no mix mp3; tracks/ is only an id registry for their
                # dataset (features come from stems), so fall back to a stem symlink
                mp3 = mp3s[0] if mp3s else stems["other"]
                if not (os.path.exists(jp) and all(os.path.exists(p) for p in stems.values())):
                    skipped += 1
                    continue
                try:
                    d = json.load(open(jp))
                    bpm = float(d["bpm"])
                    bar_len, beat_len = 240.0 / bpm, 60.0 / bpm
                    segs = sorted(((float(r["start"]), float(r["end"]), norm_label(r["name"]))
                                   for r in d["segmnets"]), key=lambda x: x[0])
                    if not segs:
                        skipped += 1
                        continue
                    ends = np.array([x["end"] for v in d["stem_elements"].values() for x in v], dtype=float)
                    ob = find_robust_onsets(ends, length=1, tolerance=0.1)
                    bar_off = float(ob[0] % 1)
                    beat_off = bar_off % 0.25
                    dur = segs[-1][1] * bar_len

                    # loader formulas: beat_k=(k+beat_off)*beat_len, bar_m=(m+bar_off)*bar_len
                    phase = int(round(4 * bar_off - beat_off)) % 4
                    with open(f"{OUT}/beats/{tid}.txt", "w") as f:
                        k = 0
                        while True:
                            t = (k + beat_off) * beat_len
                            if t > dur + 1e-6:
                                break
                            f.write(f"{t:.6f}\t{((k - phase) % 4) + 1}\n")
                            k += 1
                    rows = [(s * bar_len, l) for s, e, l in segs]
                    while rows and rows[0][1] in ("silence", None):
                        rows = rows[1:]
                    rows = [(t, l if l not in ("silence", None) else "break") for t, l in rows]
                    if not rows:
                        skipped += 1
                        continue
                    rows.append((dur, "end"))
                    with open(f"{OUT}/segments/{tid}.txt", "w") as f:
                        for t, l in rows:
                            f.write(f"{t:.6f}\t{l}\n")

                    if not os.path.exists(f"{OUT}/tracks/{tid}.mp3"):
                        os.symlink(mp3, f"{OUT}/tracks/{tid}.mp3")
                    sd = f"{OUT}/demix/htdemucs/{tid}"
                    os.makedirs(sd, exist_ok=True)
                    for s, p in stems.items():
                        if not os.path.exists(f"{sd}/{s}.wav"):
                            os.symlink(p, f"{sd}/{s}.wav")

                    meta_rows.append((tid, bpm))
                    if part == "test":
                        split["test"].append(tid)
                    elif kept % 20 == 0:
                        split["val"].append(tid)
                    else:
                        split["train"].append(tid)
                    kept += 1
                except Exception:
                    skipped += 1
    with open(f"{OUT}/metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["File", "BPM"])
        w.writerows([(t, int(round(b))) for t, b in meta_rows])
    json.dump(split, open(f"{OUT}/split.json", "w"))
    print(f"kept={kept} skipped={skipped} train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")


if __name__ == "__main__":
    main()
