"""Train phraser FROM SCRATCH on Harmonix (70/10/20 train/val/test split),
reusing the original train loop. No proprietary-corpus weights are used: only
the frozen public MuQ encoder + random init, so the result is directly
comparable to All-In-One's "trained on Harmonix" protocol.

Dataset emits the exact 5-tuple structure of CollateEmbeddingsWithPreprocess so
src.utils.train.train runs unchanged (same losses, same channel slicing;
vocals stay unsupervised). Window = 240 s (6000 frames @25 Hz).
Checkpoint selection is by val loss; the test split is touched exactly once.

Run:
  python -u -m phraser.eval.finetune_harmonix --epochs 80 --lr 2e-3
"""
import argparse
import glob
import json
import os
import random

import numpy as np

from phraser.utils.losses import label_cross_entropy
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from phraser.paths import BENCH

# PHRASER_STEM_W scales the per-stem split + silence losses (0 = disabled)
STEM_W = float(os.environ.get("PHRASER_STEM_W", "1.0"))

BASE = os.path.join(BENCH, "harmonix_ft")
SECTION_LABELS = ["silence", "intro", "outro", "bridge", "verse", "chorus"]
W = 240          # window seconds
HZ = 25          # embedding rate
BHZ = 100        # beat/onset rate


def make_split(seed=2137, test_frac=0.2, val_frac=0.1):
    path = f"{BASE}/split.json"
    if os.path.exists(path):
        s = json.load(open(path))
        if "val" in s:
            return s
    ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{BASE}/labels/*.json"))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_test = int(len(ids) * test_frac)
    n_val = int(len(ids) * val_frac)
    split = {"test": sorted(ids[:n_test]),
             "val": sorted(ids[n_test:n_test + n_val]),
             "train": sorted(ids[n_test + n_val:])}
    json.dump(split, open(path, "w"))
    return split


class HarmonixFtDataset(Dataset):
    def __init__(self, ids, train=True):
        self.ids = ids
        self.train = train

    def __len__(self):
        return len(self.ids)

    def _stem_masks(self, rms_lists, n):
        sils, splits = [], []
        for rms in rms_lists:  # order: vocals, other, bass, drums
            r = np.zeros(n, dtype=np.float32)
            m = np.asarray(rms, dtype=np.float32)[:n]
            r[: len(m)] = m
            nz = r[r > 1e-4]
            thr = max(1e-4, 0.05 * (np.median(nz) if len(nz) else 0.0))
            sil = (r < thr).astype(np.float32)
            edge = np.zeros(n, dtype=np.float32)
            d = np.flatnonzero(np.diff(sil) != 0) + 1
            edge[d] = 1.0
            sils.append(sil)
            splits.append(edge)
        return np.stack(sils), np.stack(splits)

    def __getitem__(self, idx):
        tid = self.ids[idx]
        lab = json.load(open(f"{BASE}/labels/{tid}.json"))
        emb = np.load(f"{BASE}/emb/{tid}.npy").astype(np.float32)  # [4, T, 1024]
        _mix_dir = __import__("os").environ.get("PHRASER_MIX_EMB_DIR")
        if _mix_dir:  # 5th (aggregate) input channel = MuQ of the mixture
            mix = np.load(f"{_mix_dir}/{tid}.npy").astype(np.float32)[:, :emb.shape[1]]
            mix = np.pad(mix, ((0, 0), (0, emb.shape[1] - mix.shape[1]), (0, 0)))
            emb = np.concatenate([emb, mix], axis=0)  # [5, T, 1024]
        T = emb.shape[1]
        need = W * HZ
        if T < need:
            emb = np.pad(emb, ((0, 0), (0, need - T), (0, 0)))
        dur = min(T / HZ, W)
        max_start = max(0.0, T / HZ - W)
        start = random.uniform(0.0, max_start) if (self.train and max_start > 0) else 0.0
        f0 = int(start * HZ)
        emb = emb[:, f0: f0 + need, :]

        seg_split = np.zeros(W, dtype=np.float32)
        seg_type = np.zeros((W, len(SECTION_LABELS)), dtype=np.float32)
        seg_type[:, 0] = 1.0  # default silence (also covers padding)
        segs = []
        for s, e, label in lab["segments"]:
            s2, e2 = s - start, e - start
            if e2 <= 0 or s2 >= dur:
                continue
            s2, e2 = max(s2, 0.0), min(e2, dur)
            i0, i1 = int(s2), max(int(np.ceil(e2)), int(s2) + 1)
            li = SECTION_LABELS.index(label) if label in SECTION_LABELS else SECTION_LABELS.index("verse")
            seg_type[i0:i1] = 0.0
            seg_type[i0:i1, li] = 1.0
            if 0 < e2 < dur:
                seg_split[min(int(e2 + 0.5), W - 1)] = 1.0
            segs.append((label, s2, e2))

        beats = np.zeros(W * BHZ, dtype=np.float32)
        onsets = np.zeros(W * BHZ, dtype=np.float32)
        seg_split_long = np.zeros(W * BHZ, dtype=np.float32)
        for b in lab["beats"]:
            i = int((b - start) * BHZ + 0.5)
            if 0 <= i < W * BHZ:
                beats[i] = 1.0
        for b in lab["downbeats"]:
            i = int((b - start) * BHZ + 0.5)
            if 0 <= i < W * BHZ:
                onsets[i] = 1.0
        for s, e, label in lab["segments"]:
            i = int((e - start) * BHZ + 0.5)
            if 0 < i < W * BHZ:
                seg_split_long[i] = 1.0

        n_rms = W
        f0s = int(start)
        rms_lists = [lab["stem_rms_1hz"][k][f0s:f0s + n_rms] for k in ["vocals", "other", "bass", "drums"]]
        sil, spl = self._stem_masks(rms_lists, W)
        spl_long = np.repeat(spl, BHZ, axis=1) * 0.0  # 100 Hz stem splits unused in loss; keep zeros
        for c in range(4):
            for i in np.flatnonzero(spl[c]):
                spl_long[c, min(i * BHZ, W * BHZ - 1)] = 1.0

        return {
            "embeddings": torch.from_numpy(emb).unsqueeze(0),
            "segments": segs,
            "dur": dur,
            "segments_type": torch.from_numpy(seg_type).unsqueeze(0),
            "segments_splits": torch.from_numpy(seg_split).unsqueeze(0),
            "segments_splits_long": torch.from_numpy(seg_split_long).unsqueeze(0),
            "beats": torch.from_numpy(beats).unsqueeze(0),
            "on_set": torch.from_numpy(onsets).unsqueeze(0),
            "stem_splits": torch.from_numpy(spl).unsqueeze(0),
            "stem_splits_long": torch.from_numpy(spl_long).unsqueeze(0),
            "stem_silences": torch.from_numpy(sil).unsqueeze(0),
            "path": tid,
        }


def collate(batch):
    seg_info = {
        "segments": [b["segments"] for b in batch],
        "durs": [b["dur"] for b in batch],
        "segments_type": torch.cat([b["segments_type"] for b in batch]),
        "segments_splits": torch.cat([b["segments_splits"] for b in batch]),
        "segments_splits_long": torch.cat([b["segments_splits_long"] for b in batch]),
    }
    stem_info = {
        "stem_splits": torch.cat([b["stem_splits"] for b in batch]),
        "stem_silences": torch.cat([b["stem_silences"] for b in batch]),
        "stem_splits_long": torch.cat([b["stem_splits_long"] for b in batch]),
    }
    beat_info = {
        "beats": torch.cat([b["beats"] for b in batch]),
        "on_set": torch.cat([b["on_set"] for b in batch]),
    }
    return (torch.cat([b["embeddings"] for b in batch]), seg_info, stem_info, beat_info,
            [b["path"] for b in batch])


def train_epoch_weighted(model, loader, opt, cfg, pos_weight, pw_fine=0.0):
    """Original six-term loss, but with pos_weight on the sparse binary heads
    (element/segment splits, beats, onsets) so 'never fire' stops being the
    cheap minimum on small corpora. Silence/type terms unchanged."""
    model.train()
    pw = torch.tensor([pos_weight], device="cuda")
    pwf = torch.tensor([pw_fine], device="cuda") if pw_fine > 0 else None
    for bi, (emb, seg_info, stem_info, beat_info, _paths) in enumerate(loader):
        (split_l, silent_l, sect_l), _, (beat_l, onset_l) = model.forward_encoded(emb.cuda())
        loss = (STEM_W * (F.binary_cross_entropy_with_logits(split_l[:, 1:4, :].squeeze(3), stem_info["stem_splits"][:, 1:, :].cuda().float(), pos_weight=pw)
                + F.binary_cross_entropy_with_logits(silent_l[:, 1:4, :, :].squeeze(3), stem_info["stem_silences"][:, 1:, :].cuda().float()))
                + F.binary_cross_entropy_with_logits(split_l[:, -1, :].squeeze(2), seg_info["segments_splits"].cuda().float(), pos_weight=pw)
                + label_cross_entropy(sect_l, seg_info["segments_type"].cuda()) * 0.005
                + F.binary_cross_entropy_with_logits(beat_l.squeeze(2), beat_info["beats"].cuda().float(), pos_weight=pwf)
                + F.binary_cross_entropy_with_logits(onset_l.squeeze(2), beat_info["on_set"].cuda().float(), pos_weight=pwf))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        opt.step()
        opt.zero_grad()
        if bi % 20 == 0:
            print(f"{bi}: {float(loss):.4f}", flush=True)


def train_epoch_balanced(model, loader, opt, cfg, balancer):
    """Six-term loss grouped as {label, other}; the balancer rescales each
    group's gradient to equal EMA-smoothed norm share (SongFormer's EnCodec
    balancer). Label term keeps PHRASER_LABEL_W / focal via label_cross_entropy."""
    model.train()
    params = [p for p in model.parameters_grad() if p.requires_grad]
    for bi, (emb, seg_info, stem_info, beat_info, _paths) in enumerate(loader):
        (split_l, silent_l, sect_l), _, (beat_l, onset_l) = model.forward_encoded(emb.cuda())
        label = label_cross_entropy(sect_l, seg_info["segments_type"].cuda())
        other = (STEM_W * (F.binary_cross_entropy_with_logits(split_l[:, 1:4, :].squeeze(3), stem_info["stem_splits"][:, 1:, :].cuda().float())
                 + F.binary_cross_entropy_with_logits(silent_l[:, 1:4, :, :].squeeze(3), stem_info["stem_silences"][:, 1:, :].cuda().float()))
                 + F.binary_cross_entropy_with_logits(split_l[:, -1, :].squeeze(2), seg_info["segments_splits"].cuda().float())
                 + F.binary_cross_entropy_with_logits(beat_l.squeeze(2), beat_info["beats"].cuda().float())
                 + F.binary_cross_entropy_with_logits(onset_l.squeeze(2), beat_info["on_set"].cuda().float()))
        ratios = balancer.backward({"label": label, "other": other}, params)
        torch.nn.utils.clip_grad_norm_(params, cfg.gradient_clip)
        opt.step()
        opt.zero_grad()
        if bi % 20 == 0:
            print(f"{bi}: label {float(label):.4f} other {float(other):.4f} "
                  f"gradshare label {ratios['label']:.3f}", flush=True)


def _runs_to_ints(lab):
    """Frame label sequence -> [start, end) second intervals (1 frame = 1 s)."""
    ints, labs = [], []
    s = 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            ints.append([float(s), float(i)])
            labs.append(str(lab[s]))
            s = i
    return np.asarray(ints, dtype=float), labs


def _peak_pick_1hz(prob, win=12, thr=0.5):
    """SongFormer peak_picking (postprocessing/helpers.py) rescaled to 1 Hz:
    local maxima in a win-past/win-future window, kept when the activation
    exceeds the mean of the two flanking window means (positive strength).
    Adds a 0.5 floor — our BCE split head's operating point; without it an
    untrained/flat head over-segments on noise (their head is kept flat by
    the TV loss, ours is not)."""
    from numpy.lib.stride_tricks import sliding_window_view
    L = len(prob)
    if L < 3:
        return np.zeros(L, dtype=bool)
    w = max(1, min(win, (L - 1) // 2))
    padded = np.pad(prob, (w, w), mode="constant")
    max_f = sliding_window_view(padded, 2 * w + 1)
    loc_max = (prob == max_f.max(axis=-1)) & (prob > thr)
    past = sliding_window_view(padded[: -(w + 1)], w).mean(axis=-1)
    fut = sliding_window_view(padded[w + 1:], w).mean(axis=-1)
    return loc_max & ((prob - (past + fut) / 2) > 0)


def _pooled_segments(sp, prob, dur, win=12):
    """SongFormer-style decode: boundaries from split-head peak picking only,
    segment label = argmax of the mean frame softmax inside each segment."""
    idx = np.flatnonzero(_peak_pick_1hz(sp[:dur], win))
    idx = idx[idx > 0]
    parts = np.split(prob[:dur], idx, axis=0)
    labels = [int(p.mean(axis=0).argmax()) for p in parts]
    cuts = [0] + idx.tolist() + [dur]
    ints = np.asarray([[cuts[i], cuts[i + 1]] for i in range(len(cuts) - 1)], dtype=float)
    frames = np.empty(dur, dtype=int)
    for (s, e), lb in zip(ints.astype(int), labels):
        frames[s:e] = lb
    return ints, [str(x) for x in labels], frames


class GradBalancer:
    """EnCodec-style grad-norm balancer (SongFormer train/encodec/balancer.py),
    minus distributed averaging and per-item norms (our losses are scalars).
    Each group's gradient norm is EMA-tracked; losses are rescaled so every
    group contributes weights[k]/sum(weights) of a fixed total grad norm."""

    def __init__(self, weights, total_norm=1.0, ema_decay=0.999, eps=1e-12, norm_every=1):
        self.weights = weights
        self.total_norm = total_norm
        self.ema_decay = ema_decay
        self.eps = eps
        self.norm_every = max(1, int(norm_every))
        self._num = {}
        self._den = {}
        self._step = 0
        self._scales = None
        self._ratios = None

    def backward(self, losses, params):
        # norm_every > 1: grad norms are EMA-smoothed anyway, so recompute them
        # only every Nth step and reuse the scales in between (skips the two
        # extra autograd.grad passes, ~3x cheaper per step).
        self._step += 1
        if self._scales is None or (self._step - 1) % self.norm_every == 0:
            norms = {}
            for name, loss in losses.items():
                grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                flat = torch.cat([g.reshape(-1) for g in grads if g is not None])
                norms[name] = float(flat.norm()) if flat.numel() else 0.0
            avg = {}
            for k, v in norms.items():
                self._num[k] = self._num.get(k, 0.0) * self.ema_decay + v
                self._den[k] = self._den.get(k, 0.0) * self.ema_decay + 1.0
                avg[k] = self._num[k] / self._den[k]
            tot_w = sum(self.weights[k] for k in avg)
            self._scales = {k: (self.weights[k] / tot_w) * self.total_norm / (self.eps + avg[k])
                            for k in avg}
            total = sum(avg.values())
            self._ratios = {k: v / max(total, self.eps) for k, v in avg.items()}
        out = 0
        for name, loss in losses.items():
            out = out + loss * self._scales[name]
        out.backward()
        return self._ratios


class EmaWeights:
    """EMA of trainable weights (SongFormer evaluates/checkpoints the EMA
    model only). Hooked by wrapping opt.step in main(); swap_in/out puts the
    shadow weights into the model for evaluation."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.params = [p for p in model.parameters_grad() if p.requires_grad]
        self.shadow = [p.detach().clone() for p in self.params]
        self._stash = None

    @torch.no_grad()
    def update(self):
        for s, p in zip(self.shadow, self.params):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def swap_in(self):
        self._stash = [p.detach().clone() for p in self.params]
        for p, s in zip(self.params, self.shadow):
            p.copy_(s)

    @torch.no_grad()
    def swap_out(self):
        for p, s in zip(self.params, self._stash):
            p.copy_(s)
        self._stash = None


def val_loss(model, loader, tb=None, epoch=None, tag="val"):
    """Same six loss terms as the train loop, aggregated over the val split.
    Each term is also logged as <tag>/<term> when a SummaryWriter is passed.
    Also logs <tag>/HR.5F: boundary hit-rate F at ±1 frame (≈±0.5 s at 1 Hz),
    decoded from the type head (changes) union the split head, matching
    SongFormer's HitRate_0.5 convention on the same 53 val tracks.
    Two decode variants are logged side by side:
      *_run  — per-frame argmax runs of the type head (harsh, no smoothing);
      *_pool — SongFormer-style: boundaries from split-head peak picking
               (±12 s windows), segment label = argmax of mean frame softmax.
    Plus <tag>/acc_run and <tag>/acc_pool: per-track frame label agreement."""
    model.eval()
    keys = ["elements_splits", "elements_silent", "segments_splits",
            "segments_type", "beats", "on_set"]
    tot = {k: 0.0 for k in keys}
    nb = 0
    hr_tp_g = hr_tp_p = hr_n_g = hr_n_p = 0
    pw_rows = []       # (PWP, PWR, PWF, Sf) per val track, frame-run decode
    pw_pool_rows = []  # same, SongFormer-style pooled decode
    acc_rows = []      # (acc_run, acc_pool) per val track
    try:
        import mir_eval
    except Exception:
        mir_eval = None
    with torch.no_grad():
        for emb, seg_info, stem_info, beat_info, _paths in loader:
            (split_l, silent_l, sect_l), _, (beat_l, onset_l) = model.forward_encoded(emb.cuda())
            terms = {
                "elements_splits": F.binary_cross_entropy_with_logits(split_l[:, 1:4, :].squeeze(3), stem_info["stem_splits"][:, 1:, :].cuda().float()),
                "elements_silent": F.binary_cross_entropy_with_logits(silent_l[:, 1:4, :, :].squeeze(3), stem_info["stem_silences"][:, 1:, :].cuda().float()),
                "segments_splits": F.binary_cross_entropy_with_logits(split_l[:, -1, :].squeeze(2), seg_info["segments_splits"].cuda().float()),
                "segments_type": label_cross_entropy(sect_l, seg_info["segments_type"].cuda()),
                "beats": F.binary_cross_entropy_with_logits(beat_l.squeeze(2), beat_info["beats"].cuda().float()),
                "on_set": F.binary_cross_entropy_with_logits(onset_l.squeeze(2), beat_info["on_set"].cuda().float()),
            }
            for k in keys:
                tot[k] += float(terms[k])
            nb += 1
            # HR.5F: boundary matching, tolerance ±1 frame at 1 Hz
            _sp = torch.sigmoid(split_l[:, -1, :].squeeze(2))
            _ty = sect_l.argmax(-1)
            _pred = ((_ty[:, 1:] != _ty[:, :-1]) | (_sp[:, 1:] > 0.5))
            _gt = seg_info["segments_splits"].cuda()[:, 1:] > 0.5
            _TOL = 1
            for _b in range(_pred.shape[0]):
                _pb = torch.nonzero(_pred[_b]).flatten().tolist()
                _gb = torch.nonzero(_gt[_b]).flatten().tolist()
                if not _pb and not _gb:
                    continue
                hr_n_p += len(_pb)
                hr_n_g += len(_gb)
                for _g in _gb:
                    if any(abs(_p - _g) <= _TOL for _p in _pb):
                        hr_tp_g += 1
                for _p in _pb:
                    if any(abs(_p - _g) <= _TOL for _g in _gb):
                        hr_tp_p += 1
            # PWF / Sf (mir_eval pairwise + NCE-F) on the 1 Hz frame grid:
            # est segments = runs of the type-head argmax, ref = GT argmax runs.
            if mir_eval is not None:
                _g = seg_info["segments_type"].argmax(-1).cpu().numpy()
                _e = sect_l.argmax(-1).cpu().numpy()
                _sp_np = _sp.cpu().numpy()
                _pr = torch.softmax(sect_l, -1).cpu().numpy()
                for _b in range(_g.shape[0]):
                    _d = int(min(seg_info["durs"][_b], _g.shape[1]))
                    if _d < 2:
                        continue
                    _ri, _rl = _runs_to_ints(_g[_b, :_d])
                    _ei, _el = _runs_to_ints(_e[_b, :_d])
                    _pi, _pl, _pf = _pooled_segments(_sp_np[_b], _pr[_b], _d)
                    try:
                        p, r, f = mir_eval.segment.pairwise(_ri, _rl, _ei, _el)
                        _, _, _sf = mir_eval.segment.nce(_ri, _rl, _ei, _el)
                        pw_rows.append((p, r, f, _sf))
                        p2, r2, f2 = mir_eval.segment.pairwise(_ri, _rl, _pi, _pl)
                        _, _, _sf2 = mir_eval.segment.nce(_ri, _rl, _pi, _pl)
                        pw_pool_rows.append((p2, r2, f2, _sf2))
                        acc_rows.append((float((_e[_b, :_d] == _g[_b, :_d]).mean()),
                                         float((_pf == _g[_b, :_d]).mean())))
                    except Exception:
                        pass
    model.train()
    m = {k: v / max(nb, 1) for k, v in tot.items()}
    hr_p = hr_tp_p / max(hr_n_p, 1)
    hr_r = hr_tp_g / max(hr_n_g, 1)
    hr_f = 2 * hr_p * hr_r / max(hr_p + hr_r, 1e-9)
    print(f"== {tag} HR.5F {hr_f:.4f} (P {hr_p:.4f} R {hr_r:.4f})", flush=True)
    if tb is not None:
        tb.add_scalar(f"{tag}/HR.5P", hr_p, epoch)
        tb.add_scalar(f"{tag}/HR.5R", hr_r, epoch)
        tb.add_scalar(f"{tag}/HR.5F", hr_f, epoch)
        if pw_rows:
            pwp, pwr, pwf, sfm = np.array(pw_rows).mean(0)
            print(f"== {tag} PWF {pwf:.4f} (P {pwp:.4f} R {pwr:.4f})  Sf {sfm:.4f}", flush=True)
            tb.add_scalar(f"{tag}/PWP", pwp, epoch)
            tb.add_scalar(f"{tag}/PWR", pwr, epoch)
            tb.add_scalar(f"{tag}/PWF", pwf, epoch)
            tb.add_scalar(f"{tag}/Sf", sfm, epoch)
        if pw_pool_rows:
            pwp, pwr, pwf, sfm = np.array(pw_pool_rows).mean(0)
            acr, acp = np.array(acc_rows).mean(0)
            print(f"== {tag} PWF_pool {pwf:.4f} (P {pwp:.4f} R {pwr:.4f})  Sf_pool {sfm:.4f}"
                  f"  acc_run {acr:.4f}  acc_pool {acp:.4f}", flush=True)
            tb.add_scalar(f"{tag}/PWP_pool", pwp, epoch)
            tb.add_scalar(f"{tag}/PWR_pool", pwr, epoch)
            tb.add_scalar(f"{tag}/PWF_pool", pwf, epoch)
            tb.add_scalar(f"{tag}/Sf_pool", sfm, epoch)
            tb.add_scalar(f"{tag}/acc_run", acr, epoch)
            tb.add_scalar(f"{tag}/acc_pool", acp, epoch)
        for k, v in m.items():
            tb.add_scalar(f"{tag}/{k}", v, epoch)
    return sum(v * (STEM_W if k in ("elements_splits", "elements_silent") else 1.0) for k, v in m.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--init-ckpt", default=None,
                    help="optional warm start; default is FROM SCRATCH (random init)")
    ap.add_argument("--pw-fine", type=float, default=0.0, help="pos_weight for 100Hz heads (0=plain BCE)")
    ap.add_argument("--pos-weight", type=float, default=0.0,
                    help=">0 enables pos_weight on sparse binary heads (small-corpus rescue)")
    ap.add_argument("--out", default=f"{BASE}/scratch_run")
    ap.add_argument("--seed", type=int, default=2137,
                    help="torch/numpy/random seed (init, dropout, shuffling); split is file-fixed")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from phraser.modules.phraser import PhraserModel
    from phraser.utils.train import train, cosine_lr

    split = make_split()
    print(f"train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}", flush=True)
    tr_dl = DataLoader(HarmonixFtDataset(split["train"], train=True), batch_size=args.batch_size,
                       shuffle=True, collate_fn=collate, num_workers=6, drop_last=True)
    va_dl = DataLoader(HarmonixFtDataset(split["val"], train=False), batch_size=args.batch_size,
                       shuffle=False, collate_fn=collate, num_workers=4, drop_last=False)

    model = PhraserModel(128, 16)
    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
        model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("muq.")}, strict=False)
        print(f"warm start from {args.init_ckpt}", flush=True)
    else:
        print("training FROM SCRATCH (random init, frozen public MuQ features only)", flush=True)
    model = model.cuda()

    cfg = argparse.Namespace(distributed=False, is_master=True, gradient_clip=1.0,
                             lr=args.lr, warmup=args.warmup, epochs=args.epochs, world_size=1)
    opt = torch.optim.AdamW(model.parameters_grad(), lr=args.lr, weight_decay=1e-2,
                            betas=(0.9, 0.95), eps=1e-8)
    sched = cosine_lr(opt, args.lr, args.warmup, args.epochs)

    os.makedirs(f"{args.out}/checkpoints", exist_ok=True)
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(log_dir=os.environ.get("PHRASER_TB_DIR", f"{args.out}/tb"))
    balancer = None
    if os.environ.get("PHRASER_GRAD_BAL", "0") == "1":
        w_label = float(os.environ.get("PHRASER_BAL_W_LABEL", "1.0"))
        n_every = int(os.environ.get("PHRASER_BAL_NORM_EVERY", "1"))
        balancer = GradBalancer({"label": w_label, "other": 1.0}, norm_every=n_every)
        print(f"GRAD_BAL on (label grad share {w_label / (w_label + 1.0):.2f}, norm_every {n_every})", flush=True)
    ema = None
    if os.environ.get("PHRASER_EMA", "0") == "1":
        ema = EmaWeights(model, float(os.environ.get("PHRASER_EMA_DECAY", "0.999")))
        _orig_step = opt.step

        def _step_and_update(*a, **k):
            r = _orig_step(*a, **k)
            ema.update()
            return r

        opt.step = _step_and_update
        print(f"EMA on (decay {ema.decay}, per-step; val_ema/* logged)", flush=True)
    best = float("inf")
    for epoch in range(args.epochs):
        sched(epoch)
        print(f"== epoch {epoch} (lr {opt.param_groups[0]['lr']:.2e})", flush=True)
        if balancer is not None:
            train_epoch_balanced(model, tr_dl, opt, cfg, balancer)
        elif args.pos_weight > 0:
            train_epoch_weighted(model, tr_dl, opt, cfg, args.pos_weight, args.pw_fine)
        else:
            train(model, tr_dl, opt, epoch, "cuda", loggers=[], cfg=cfg)
        vl = val_loss(model, va_dl, tb, epoch, tag="val")
        tb.add_scalar("val/loss", vl, epoch)
        if ema is not None:
            ema.swap_in()
            vl_e = val_loss(model, va_dl, tb, epoch, tag="val_ema")
            tb.add_scalar("val_ema/loss", vl_e, epoch)
            ema.swap_out()
            print(f"== epoch {epoch} val_loss {vl:.4f}  val_ema_loss {vl_e:.4f}", flush=True)
        else:
            print(f"== epoch {epoch} val_loss {vl:.4f}", flush=True)
        if vl < best:
            best = vl
            model.save(args.out, cfg, "best", epoch=epoch, val_loss=vl)
            print(f"== new best (epoch {epoch})", flush=True)
        if epoch % 10 == 0:
            model.save(args.out, cfg, f"epoch_{epoch}", epoch=epoch)
    model.save(args.out, cfg, "final", epoch=args.epochs - 1)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
