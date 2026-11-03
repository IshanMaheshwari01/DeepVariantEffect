"""Feature construction for the logistic-regression baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from dve.features import center_crop, delta_kmer_counts, encode, kmer_counts, trinuc_class


def logreg_features(df: pd.DataFrame, cfg: dict) -> sparse.csr_matrix:
    """[96 substitution classes | context k-mer profile | alt-ref delta k-mers]."""
    flank = cfg["flank_in"]
    ref_seqs = df["ref_seq"].tolist()
    alt_seqs = df["alt_seq"].tolist()

    # 1) trinucleotide substitution class, one-hot (96)
    tri = np.array([trinuc_class(s, a, flank) for s, a in zip(ref_seqs, df["alt"], strict=True)])
    tri_ok = tri >= 0
    tri_mat = sparse.csr_matrix(
        (np.ones(tri_ok.sum()), (np.flatnonzero(tri_ok), tri[tri_ok])), shape=(len(df), 96)
    )

    # 2) local sequence composition: k-mer frequencies of the reference context
    ctx = encode(center_crop(ref_seqs, flank, cfg["context_flank"]))
    ctx_counts = kmer_counts(ctx, cfg["context_k"])
    ctx_freq = ctx_counts / np.maximum(ctx_counts.sum(axis=1, keepdims=True), 1)
    ctx_freq = (ctx_freq - ctx_freq.mean(axis=0)) / (ctx_freq.std(axis=0) + 1e-6)

    # 3) which k-mers the variant destroys / creates
    # built in chunks: one dense matrix of all variants x 4096 k-mers would need about 7 GB
    delta = sparse.vstack(
        [
            sparse.csr_matrix(
                delta_kmer_counts(
                    ref_seqs[i : i + 5000], alt_seqs[i : i + 5000], flank, cfg["delta_k"]
                )
            )
            for i in range(0, len(ref_seqs), 5000)
        ],
        format="csr",
    )

    return sparse.hstack(
        [tri_mat, sparse.csr_matrix(ctx_freq), sparse.csr_matrix(delta)], format="csr"
    )
