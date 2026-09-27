"""Build the data layout for retraining All-In-One on our Harmonix audio.

Creates $PHRASER_DATA/benchmarks/allin1_train/ with:
  tracks/<id>.mp3            symlinks to harmonix_audio (only alignment-verified tracks)
  beats/<id>.txt             offset/rate-corrected beat+downbeat annotations (audio timeline)
  segments/<id>.txt          offset/rate-corrected segment annotations
  metadata.csv               copy
Correction: t_audio = (t_ref - shift) / rate from harmonix_alignment_fits.json —
the same shared per-track alignment used in the comparison study. Tracks failing
the 70 ms gate (frac70 < 0.7) are excluded from training data entirely.
"""
import csv
import json
import os
import shutil
from phraser.paths import BENCH

BASE = BENCH
OUT = f"{BASE}/allin1_train"

ALLIN1_VOCAB = {"start", "end", "intro", "outro", "break", "bridge", "inst", "solo", "verse", "chorus"}
LABEL_FIX = {
    "prechorus": "verse", "postchorus": "chorus", "altchorus": "chorus", "instchorus": "chorus",
    "instrumental": "inst", "gtr": "inst", "section": "inst", "transition": "inst",
    "breakdown": "break", "quiet": "break", "refrain": "chorus", "raps": "verse",
    "bre": "break", "gtrbreak": "break", "guitar": "inst", "vocaloutro": "outro",
}


def normalize_label(raw):
    l = raw.strip().lower().rstrip("0123456789")
    if l == "silence":
        return "silence"
    l = LABEL_FIX.get(l, l)
    return l if l in ALLIN1_VOCAB else "inst"


def main():
    fits = json.load(open(f"{BASE}/harmonix_alignment_fits.json"))
    for sub in ["tracks", "beats", "segments"]:
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)

    kept = skipped = 0
    for tid, f in fits.items():
        if f.get("frac70", 0) < 0.7:
            skipped += 1
            continue
        audio = f"{BASE}/harmonix_audio/{tid}.mp3"
        if not os.path.exists(audio):
            continue
        rate, shift = f["rate"], f["shift"]

        def to_audio(t):
            return (t - shift) / rate

        link = f"{OUT}/tracks/{tid}.mp3"
        if not os.path.exists(link):
            os.symlink(audio, link)

        # their reader: 2 columns (time, beat-in-bar), tab-separated
        with open(f"{BASE}/harmonixset/dataset/beats_and_downbeats/{tid}.txt") as fin, \
                open(f"{OUT}/beats/{tid}.txt", "w") as fout:
            for line in fin:
                parts = line.split()
                if len(parts) >= 2:
                    t = to_audio(float(parts[0]))
                    if t >= 0:
                        fout.write(f"{t:.6f}\t{parts[1]}\n")

        # their vocab: start/end/intro/outro/break/bridge/inst/solo/verse/chorus
        # ('start' is prepended by their converter; file must end with 'end')
        rows = []
        for line in open(f"{BASE}/harmonixset/dataset/segments/{tid}.txt"):
            parts = line.strip().split()
            if len(parts) >= 2:
                rows.append((max(to_audio(float(parts[0])), 0.0), normalize_label(parts[1])))
        while rows and rows[0][1] in ("silence", None):  # leading silence -> covered by 'start'
            rows = rows[1:]
        rows = [(t, l if l not in ("silence", None) else "break") for t, l in rows]
        if rows and rows[-1][1] != "end":
            rows.append((rows[-1][0] + 0.1, "end"))
        with open(f"{OUT}/segments/{tid}.txt", "w") as fout:
            for t, l in rows:
                fout.write(f"{t:.6f}\t{l}\n")
        kept += 1

    shutil.copy(f"{BASE}/harmonixset/dataset/metadata.csv", f"{OUT}/metadata.csv")
    print(f"setup done: kept={kept} skipped_gate={skipped}")


if __name__ == "__main__":
    main()
