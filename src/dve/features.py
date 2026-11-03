"""Sequence encodings and hand-crafted features shared by all models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

BASES = "ACGT"
_LUT = np.full(256, 4, dtype=np.uint8)  # anything not ACGT -> 4
for _i, _b in enumerate(BASES):
    _LUT[ord(_b)] = _i
_COMP = str.maketrans("ACGT", "TGCA")


def encode(seqs: Sequence[str]) -> np.ndarray:
    """Encode equal-length DNA strings as an (N, L) uint8 array with A=0 C=1 G=2 T=3."""
    if len(seqs) == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    length = len(seqs[0])
    buf = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8)
    return _LUT[buf].reshape(len(seqs), length)


def center_crop(seqs: Sequence[str], flank_in: int, flank_out: int) -> list[str]:
    """Cut a (2*flank_in+1) window down to (2*flank_out+1) around the centre."""
    if flank_out > flank_in:
        raise ValueError("flank_out must be <= flank_in")
    a, b = flank_in - flank_out, flank_in + flank_out + 1
    return [s[a:b] for s in seqs]


# --- 96-class substitution spectrum (like COSMIC mutational signatures) ---------
_SUBS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
_TRINUC_INDEX = {
    f"{five}[{sub}]{three}": i
    for i, (sub, five, three) in enumerate(
        (sub, five, three) for sub in _SUBS for five in BASES for three in BASES
    )
}


def trinuc_class(ref_seq: str, alt: str, flank: int) -> int:
    """Index 0-95 of the pyrimidine-collapsed trinucleotide substitution."""
    left, ref, right = ref_seq[flank - 1], ref_seq[flank], ref_seq[flank + 1]
    if ref in "GA":  # collapse onto the pyrimidine strand
        left, ref, right = right.translate(_COMP), ref.translate(_COMP), left.translate(_COMP)
        alt = alt.translate(_COMP)
    return _TRINUC_INDEX.get(f"{left}[{ref}>{alt}]{right}", -1)


def kmer_counts(enc: np.ndarray, k: int) -> np.ndarray:
    """Count all k-mers per row of an encoded (N, L) array -> (N, 4**k) float32.

    k-mers containing an unknown base (code 4) are ignored.
    """
    n, length = enc.shape
    n_kmers = length - k + 1
    idx = np.zeros((n, n_kmers), dtype=np.int64)
    valid = np.ones((n, n_kmers), dtype=bool)
    for j in range(k):
        col = enc[:, j : j + n_kmers].astype(np.int64)
        valid &= col < 4
        idx = idx * 4 + np.minimum(col, 3)
    rows = np.repeat(np.arange(n), n_kmers).reshape(n, n_kmers)
    flat = (rows * 4**k + idx)[valid]
    counts = np.bincount(flat, minlength=n * 4**k).reshape(n, 4**k)
    return counts.astype(np.float32)


def delta_kmer_counts(ref_seqs: Sequence[str], alt_seqs: Sequence[str], flank: int, k: int):
    """alt - ref k-mer counts. Only the 2k-1 bases around the variant can differ."""
    a, b = flank - (k - 1), flank + k
    ref_enc = encode([s[a:b] for s in ref_seqs])
    alt_enc = encode([s[a:b] for s in alt_seqs])
    return kmer_counts(alt_enc, k) - kmer_counts(ref_enc, k)


def one_hot_pair(ref_enc: np.ndarray, alt_enc: np.ndarray) -> np.ndarray:
    """Stack ref one-hot (4) + alt one-hot (4) + centre marker (1) -> (N, 9, L) float32."""
    n, length = ref_enc.shape
    eye = np.vstack([np.eye(4, dtype=np.float32), np.zeros((1, 4), dtype=np.float32)])
    ref_oh = eye[ref_enc].transpose(0, 2, 1)
    alt_oh = eye[alt_enc].transpose(0, 2, 1)
    marker = np.zeros((n, 1, length), dtype=np.float32)
    marker[:, 0, length // 2] = 1.0
    return np.concatenate([ref_oh, alt_oh, marker], axis=1)
