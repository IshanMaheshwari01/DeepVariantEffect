"""Reference-genome access and sequence-context extraction."""

from __future__ import annotations

from pyfaidx import Fasta


class Reference:
    """Thin wrapper around pyfaidx for fast random access to GRCh38."""

    def __init__(self, fasta_path: str):
        self._fa = Fasta(fasta_path, as_raw=True, sequence_always_upper=True)

    def chrom_length(self, chrom: str) -> int:
        return len(self._fa[chrom])

    def window(self, chrom: str, pos: int, flank: int) -> str | None:
        """Return the reference window centred on a 1-based position.

        The window has length 2 * flank + 1 and the variant base sits at
        index `flank`. Returns None if the window runs off the chromosome.
        """
        if chrom not in self._fa:
            return None
        center = pos - 1  # 0-based
        start, end = center - flank, center + flank + 1
        if start < 0 or end > self.chrom_length(chrom):
            return None
        return self._fa[chrom][start:end]


def make_alt(ref_seq: str, flank: int, alt: str) -> str:
    """Substitute the centre base of a reference window with the alt allele."""
    return ref_seq[:flank] + alt + ref_seq[flank + 1 :]
