"""Loss functions with explicit axis conventions.

The label head emits logits shaped [B, T, C] (time-major, C section
classes). ``torch.nn.functional.cross_entropy`` takes the CLASS axis as
dim 1, so passing that tensor directly makes the softmax run over TIME
while the one-hot float targets keep the call shape-valid --- a silent
mis-specification this module exists to prevent (the released paper
checkpoints were trained with that behavior; see
tests/test_label_loss.py for the regression and canary tests).
"""
import torch
import torch.nn.functional as F

MAX_CLASSES = 32  # the class axis is small; time is not


def label_cross_entropy(logits, target_onehot, weight=None, label_smoothing=0.0):
    """Mean class-axis cross-entropy for [..., T, C] label logits.

    ``target_onehot`` has the same shape; all-zero rows (unlabeled padding)
    are masked out. Returns a scalar that is ~log(C) at random init and
    independent of sequence length by construction.
    """
    if logits.shape != target_onehot.shape:
        raise ValueError(
            f"shape mismatch: {tuple(logits.shape)} vs {tuple(target_onehot.shape)}")
    C = logits.shape[-1]
    if C > MAX_CLASSES:
        raise ValueError(
            f"last dim is {C}; expected the (small) class axis last, time before it")
    lg = logits.reshape(-1, C)
    oh = target_onehot.reshape(-1, C)
    mask = oh.sum(-1) > 0
    if not torch.any(mask):
        return lg.sum() * 0.0
    return F.cross_entropy(lg[mask], oh[mask].argmax(-1),
                           weight=weight, label_smoothing=label_smoothing)
