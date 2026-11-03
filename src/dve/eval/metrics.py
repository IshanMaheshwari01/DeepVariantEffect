"""Evaluation metrics with bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def evaluate(y_true, y_score, n_boot: int = 200, seed: int = 0) -> dict:
    """AUROC, AUPRC and Brier score, with 95% bootstrap CIs for AUROC and AUPRC."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    out = {
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, np.clip(y_score, 0, 1))),
    }
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        aurocs, auprcs = [], []
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            yt, ys = y_true[idx], y_score[idx]
            if yt.min() == yt.max():
                continue
            aurocs.append(roc_auc_score(yt, ys))
            auprcs.append(average_precision_score(yt, ys))
        out["auroc_ci95"] = [float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))]
        out["auprc_ci95"] = [float(np.percentile(auprcs, 2.5)), float(np.percentile(auprcs, 97.5))]
    return out
