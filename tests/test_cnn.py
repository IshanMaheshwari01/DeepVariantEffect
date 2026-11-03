"""CNN shape and reverse-complement tests (skipped if torch is not installed, e.g. in CI)."""

import pytest

torch = pytest.importorskip("torch")

from dve.models.cnn import VariantCNN, reverse_complement  # noqa: E402


def test_forward_shape():
    model = VariantCNN(channels=16)
    x = torch.zeros(4, 9, 201)
    assert model(x).shape == (4,)


def test_reverse_complement_involution():
    x = torch.rand(2, 9, 11)
    assert torch.allclose(reverse_complement(reverse_complement(x)), x)


def test_reverse_complement_swaps_bases():
    x = torch.zeros(1, 9, 3)
    x[0, 0, 0] = 1  # ref A at position 0
    rc = reverse_complement(x)
    assert rc[0, 3, 2] == 1  # becomes ref T at the last position
