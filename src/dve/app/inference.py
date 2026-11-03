"""Scoring for the demo, using the CNN (small enough to run without a GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dve.features import center_crop, encode, one_hot_pair
from dve.models.cnn import VariantCNN, reverse_complement

WINDOW, MID, CNN_FLANK = 1025, 512, 100
CKPT = "outputs/cnn_onehot/best.pt"


def load_model(path=CKPT, channels=128, dropout=0.3) -> VariantCNN:
    m = VariantCNN(channels=channels, dropout=dropout)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


def platt(path="models/app_meta.json") -> tuple[float, float]:
    meta = json.loads(Path(path).read_text())
    return meta["a"], meta["b"]


def _inputs(ref, alt) -> torch.Tensor:
    crop = lambda q: encode(center_crop(q, MID, CNN_FLANK))  # noqa: E731
    return torch.from_numpy(one_hot_pair(crop(ref), crop(alt)))


@torch.no_grad()
def logits(model, ref: list[str], alt: list[str], bs: int = 64) -> np.ndarray:
    """Logits averaged over both strands, as in training."""
    x = _inputs(ref, alt)
    out = [
        (model(x[i : i + bs]) + model(reverse_complement(x[i : i + bs]))) / 2
        for i in range(0, len(x), bs)
    ]
    return torch.cat(out).numpy()


def probability(logit, a: float, b: float) -> np.ndarray:
    return 1 / (1 + np.exp(-(a * np.asarray(logit) + b)))


def saliency(model, ref_seq: str, alt_seq: str, span: int = 30) -> np.ndarray:
    """In-silico mutagenesis. Change each base near the variant to each other base, in ref
    and alt together, and record how far the logit moves. Returns (4, 2 * span + 1) with
    zeros at the variant and at each position's own base. span must not exceed CNN_FLANK."""
    base = logits(model, [ref_seq], [alt_seq])[0]
    rs, ass, where = [], [], []
    for p in range(MID - span, MID + span + 1):
        if p == MID:
            continue
        for i, b in enumerate("ACGT"):
            if b != ref_seq[p]:
                rs.append(ref_seq[:p] + b + ref_seq[p + 1 :])
                ass.append(alt_seq[:p] + b + alt_seq[p + 1 :])
                where.append((i, p - (MID - span)))
    delta = logits(model, rs, ass) - base
    out = np.zeros((4, 2 * span + 1), dtype=np.float32)
    for (i, j), d in zip(where, delta, strict=True):
        out[i, j] = d
    return out


def make_alt_seq(ref_seq: str, alt: str) -> str:
    return ref_seq[:MID] + alt + ref_seq[MID + 1 :]


def validate_window(seq: str) -> str | None:
    """Returns an error message, or None if the sequence is usable."""
    if len(seq) != WINDOW:
        return f"Need exactly {WINDOW} bases, got {len(seq)}."
    if set(seq) - set("ACGT"):
        return "Use only A, C, G and T."
    return None
