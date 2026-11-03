"""Metrics shared by every model. Confidence intervals resample genes, not variants,
because variants in one gene are correlated and would otherwise give CIs that are too narrow."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def auroc(y, s) -> float:
    return float(roc_auc_score(y, s))


def auprc(y, s) -> float:
    return float(average_precision_score(y, s))


def _gene_groups(genes) -> list[np.ndarray]:
    codes = np.unique(np.asarray(genes, dtype=object).astype(str), return_inverse=True)[1]
    order = np.argsort(codes, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)


def bootstrap_ci(y, s, genes, fn=auroc, n_boot=500, seed=0) -> tuple[float, float]:
    """95% percentile CI from a bootstrap over genes."""
    y, s = np.asarray(y), np.asarray(s)
    groups = _gene_groups(genes)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        if y[idx].min() != y[idx].max():
            stats.append(fn(y[idx], s[idx]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_delta_ci(y, a, b, genes, fn=auroc, n_boot=500, seed=0) -> tuple[float, float, float]:
    """fn(a) - fn(b) on the same variants, with a gene-bootstrap 95% CI."""
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    groups = _gene_groups(genes)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        if y[idx].min() != y[idx].max():
            deltas.append(fn(y[idx], a[idx]) - fn(y[idx], b[idx]))
    return (
        float(fn(y, a) - fn(y, b)),
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
    )


def within_gene_auroc(y, s, genes, min_each=5) -> tuple[float, int]:
    """Mean AUROC inside genes with at least `min_each` variants of each class.

    Shows whether a score separates variants in the same gene, which is the clinically
    useful question, rather than just telling constrained genes from tolerant ones.
    """
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for g in _gene_groups(genes):
        pos = int(y[g].sum())
        if pos >= min_each and len(g) - pos >= min_each:
            vals.append(auroc(y[g], s[g]))
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt_calibrate(y_val, s_val, s_test) -> np.ndarray:
    """Fit Platt scaling on validation scores and apply it to test scores."""
    lr = LogisticRegression(C=1e6)
    lr.fit(_logit(np.asarray(s_val)).reshape(-1, 1), y_val)
    return lr.predict_proba(_logit(np.asarray(s_test)).reshape(-1, 1))[:, 1]


def ece(y, p, bins=10) -> float:
    """Expected calibration error with equal-mass bins."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p)
    return float(
        sum(
            len(c) / len(y) * abs(y[c].mean() - p[c].mean())
            for c in np.array_split(order, bins)
            if len(c)
        )
    )


def reliability(y, p, bins=10) -> tuple[np.ndarray, np.ndarray]:
    y, p = np.asarray(y), np.asarray(p)
    parts = [c for c in np.array_split(np.argsort(p), bins) if len(c)]
    return np.array([p[c].mean() for c in parts]), np.array([y[c].mean() for c in parts])


def summarize(y_test, s_test, genes_test, y_val=None, s_val=None, n_boot=500) -> dict:
    """The full metric block saved for each model."""
    out = {
        "n": int(len(y_test)),
        "n_pathogenic": int(np.sum(y_test)),
        "prevalence": float(np.mean(y_test)),
        "auroc": auroc(y_test, s_test),
        "auprc": auprc(y_test, s_test),
        "auroc_ci": bootstrap_ci(y_test, s_test, genes_test, auroc, n_boot),
        "auprc_ci": bootstrap_ci(y_test, s_test, genes_test, auprc, n_boot),
    }
    out["within_gene_auroc"], out["within_gene_n_genes"] = within_gene_auroc(
        y_test, s_test, genes_test
    )
    if y_val is not None:
        p = platt_calibrate(y_val, s_val, s_test)
        out["brier"] = float(brier_score_loss(y_test, p))
        out["brier_baseline"] = float(brier_score_loss(y_test, np.full(len(p), np.mean(y_val))))
        out["ece"] = ece(y_test, p)
    return out
