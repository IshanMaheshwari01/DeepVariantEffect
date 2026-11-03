"""Generic training helpers used by the CNN and the transformer."""

from __future__ import annotations

import numpy as np
import torch


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pos_weight(y: np.ndarray, dev) -> torch.Tensor:
    return torch.tensor((len(y) - y.sum()) / max(y.sum(), 1), dtype=torch.float32, device=dev)


class EarlyStopper:
    def __init__(self, patience: int):
        self.patience, self.best, self.bad = patience, -1.0, 0

    def step(self, value: float) -> tuple[bool, bool]:
        """Returns (improved, should_stop)."""
        if value > self.best:
            self.best, self.bad = value, 0
            return True, False
        self.bad += 1
        return False, self.bad >= self.patience
