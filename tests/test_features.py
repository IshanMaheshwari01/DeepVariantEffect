"""Tests for sequence features and metrics (no torch needed)."""

import numpy as np
import pandas as pd

from dve.eval.metrics import evaluate
from dve.features import (
    center_crop,
    delta_kmer_counts,
    encode,
    kmer_counts,
    one_hot_pair,
    trinuc_class,
)
from dve.models.baselines import logreg_features


def test_encode():
    enc = encode(["ACGT", "TTNA"])
    assert enc.tolist() == [[0, 1, 2, 3], [3, 3, 4, 0]]


def test_center_crop():
    assert center_crop(["AAACGTTT"[:7]], 3, 1) == ["ACG"]


def test_kmer_counts_simple():
    counts = kmer_counts(encode(["AAAA"]), 2)
    assert counts.shape == (1, 16)
    assert counts[0, 0] == 3  # "AA" appears 3 times
    assert counts.sum() == 3


def test_kmer_counts_skips_n():
    assert kmer_counts(encode(["ANA"]), 2).sum() == 0


def test_delta_kmer_counts():
    ref, alt = "AAAAAAA", "AAACAAA"
    d = delta_kmer_counts([ref], [alt], flank=3, k=2)
    assert d.sum() == 0  # same number of k-mers lost and gained
    assert d[0, 0] == -2  # two "AA" lost
    assert d[0, 1] == 1 and d[0, 4] == 1  # "AC" and "CA" gained


def test_trinuc_class_strand_collapse():
    # A[C>T]G on + strand equals C[G>A]T on - strand -> same class
    assert trinuc_class("ACG", "T", 1) == trinuc_class("CGT", "A", 1)
    assert 0 <= trinuc_class("ACG", "T", 1) < 96


def test_one_hot_pair_shape():
    x = one_hot_pair(encode(["ACGTA"]), encode(["ACTTA"]))
    assert x.shape == (1, 9, 5)
    assert x[0, 8].tolist() == [0, 0, 1, 0, 0]  # centre marker
    assert x[0, :4].sum() == 5 and x[0, 4:8].sum() == 5


def test_logreg_features_shape():
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 21)) for _ in range(5)]
    alts = ["A" if s[10] != "A" else "C" for s in seqs]
    alt_seqs = [s[:10] + a + s[11:] for s, a in zip(seqs, alts, strict=True)]
    df = pd.DataFrame({"ref_seq": seqs, "alt": alts, "alt_seq": alt_seqs})
    cfg = {"flank_in": 10, "context_flank": 5, "context_k": 2, "delta_k": 3}
    X = logreg_features(df, cfg)
    assert X.shape == (5, 96 + 16 + 64)


def test_evaluate():
    y = np.array([0, 0, 1, 1] * 25)
    s = np.array([0.1, 0.2, 0.8, 0.9] * 25)
    r = evaluate(y, s, n_boot=20)
    assert r["auroc"] == 1.0 and r["auprc"] == 1.0
    assert r["auroc_ci95"][0] <= r["auroc"] <= r["auroc_ci95"][1]
