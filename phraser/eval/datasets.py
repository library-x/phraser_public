"""Benchmark annotation adapters: Harmonix Set and SALAMI.

Both return lists of dicts: {id, audio (path or None), ref_intervals, ref_labels, ref_beats}.
Label vocabularies are mapped onto the model's six functions via LABEL_MAP;
unmapped labels fall back to 'verse' (same convention as training's get_index).
"""
import csv
import glob
import os
import numpy as np

# model vocabulary: silence, intro, outro, bridge, verse, chorus
LABEL_MAP = {
    "silence": "silence", "intro": "intro", "outro": "outro", "end": "outro",
    "bridge": "bridge", "verse": "verse", "chorus": "chorus", "refrain": "chorus",
    "pre-chorus": "verse", "prechorus": "verse", "post-chorus": "chorus", "postchorus": "chorus",
    "solo": "bridge", "instrumental": "bridge", "inst": "bridge", "break": "bridge",
    "breakdown": "bridge", "interlude": "bridge", "transition": "bridge",
    "head": "verse", "main theme": "verse", "theme": "verse",
}


def map_label(raw):
    return LABEL_MAP.get(str(raw).strip().lower().rstrip("0123456789_ "), "verse")


def load_harmonix(root, audio_dir=None):
    """root = clone of github.com/urinieto/harmonixset (dataset/ inside).

    Segments: dataset/segments/<id>.txt lines '<time>\t<label>'.
    Beats:    dataset/beats_and_downbeats/<id>.txt lines '<time>\t<beat>\t<bar>'.
    Audio is NOT distributed; pass audio_dir with '<id>.<ext>' files if available.
    """
    tracks = []
    for seg_path in sorted(glob.glob(os.path.join(root, "dataset", "segments", "*.txt"))):
        tid = os.path.splitext(os.path.basename(seg_path))[0]
        times, labels = [], []
        with open(seg_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    times.append(float(parts[0]))
                    labels.append(map_label(" ".join(parts[1:])))
        if len(times) < 2:
            continue
        intervals = np.array([[times[i], times[i + 1]] for i in range(len(times) - 1)])
        seg_labels = labels[:-1]
        beats = []
        bpath = os.path.join(root, "dataset", "beats_and_downbeats", tid + ".txt")
        if os.path.exists(bpath):
            with open(bpath) as f:
                beats = [float(l.split()[0]) for l in f if l.strip()]
        audio = None
        if audio_dir:
            hits = glob.glob(os.path.join(audio_dir, tid + ".*"))
            audio = hits[0] if hits else None
        tracks.append({"id": tid, "audio": audio, "ref_intervals": intervals,
                       "ref_labels": seg_labels, "ref_beats": np.array(beats)})
    return tracks


def load_salami(root, audio_dir=None, annotator="textfile1"):
    """root = clone of github.com/DDMAL/salami-data-public.

    Uses parsed 'uppercase' functional files:
    annotations/<id>/parsed/<annotator>_functions.txt lines '<time>\t<label>'.
    audio_dir: '<id>.<ext>' files (e.g. the Internet Archive subset).
    """
    tracks = []
    for d in sorted(glob.glob(os.path.join(root, "annotations", "*")), key=lambda p: p):
        tid = os.path.basename(d)
        fpath = os.path.join(d, "parsed", f"{annotator}_functions.txt")
        if not os.path.exists(fpath):
            continue
        times, labels = [], []
        with open(fpath) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    times.append(float(parts[0]))
                    labels.append(map_label(parts[1]))
        if len(times) < 2:
            continue
        intervals = np.array([[times[i], times[i + 1]] for i in range(len(times) - 1)])
        audio = None
        if audio_dir:
            hits = glob.glob(os.path.join(audio_dir, tid + ".*"))
            audio = hits[0] if hits else None
        tracks.append({"id": tid, "audio": audio, "ref_intervals": intervals,
                       "ref_labels": labels[:-1], "ref_beats": np.array([])})
    return tracks


def load_bhx(root, audio_dir=None):
    """SongFormBench-HX-200: labels re-aligned to the bench's own audio.

    root = .../songformbench/data/labels/HarmonixSet (BHX_<id>.txt, '<time> <label>'
    rows ending with 'end'); audio_dir = .../data/audios/HarmonixSet (BHX_<id>.wav).
    No offset fits apply: audio and labels are self-consistent by construction.
    """
    tracks = []
    for f in sorted(glob.glob(os.path.join(root, "BHX_*.txt"))):
        tid = os.path.basename(f)[4:-4]
        times, labels = [], []
        with open(f) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2:
                    times.append(float(parts[0]))
                    labels.append(map_label(parts[1]))
        if len(times) < 2:
            continue
        intervals = np.array([[times[i], times[i + 1]] for i in range(len(times) - 1)])
        audio = None
        if audio_dir:
            cand = os.path.join(audio_dir, f"BHX_{tid}.wav")
            audio = cand if os.path.exists(cand) else None
        tracks.append({"id": tid, "audio": audio, "ref_intervals": intervals,
                       "ref_labels": labels[:-1], "ref_beats": np.array([])})
    return tracks
