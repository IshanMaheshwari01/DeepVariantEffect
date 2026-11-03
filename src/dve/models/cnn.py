"""A compact 1D CNN that reads the reference and alternate sequence together."""

from __future__ import annotations

import torch
from torch import nn


def _block(c_in: int, c_out: int, k: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(inplace=True),
    )


class VariantCNN(nn.Module):
    """Input (N, 9, L): ref one-hot, alt one-hot, centre marker. Output: logits (N,)."""

    def __init__(self, in_channels: int = 9, channels: int = 128, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            _block(in_channels, channels, 15),
            nn.MaxPool1d(2),
            _block(channels, channels, 9),
            nn.MaxPool1d(2),
            _block(channels, channels, 5),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(3 * channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        centre = h[:, :, h.shape[-1] // 2]  # features at the variant position
        pooled = torch.cat([h.amax(dim=-1), h.mean(dim=-1), centre], dim=1)
        return self.head(pooled).squeeze(-1)


def reverse_complement(x: torch.Tensor) -> torch.Tensor:
    """Reverse-complement a (N, 9, L) batch: flip length, swap A<->T and C<->G."""
    x = x.flip(-1)
    perm = [3, 2, 1, 0, 7, 6, 5, 4, 8]
    return x[:, perm, :]
