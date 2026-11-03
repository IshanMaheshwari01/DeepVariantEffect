"""Collect all model results into one table and one ROC / PR figure.

Usage:
    uv run python scripts/compare_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402


def results_table(results_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(results_dir.glob("*.json")):
        r = json.loads(f.read_text())
        if "test" not in r:
            continue
        t = r["test"]
        auroc_ci = f"[{t['auroc_ci95'][0]:.3f}-{t['auroc_ci95'][1]:.3f}]"
        auprc_ci = f"[{t['auprc_ci95'][0]:.3f}-{t['auprc_ci95'][1]:.3f}]"
        rows.append(
            {
                "model": r["model"],
                "test AUROC": f"{t['auroc']:.3f} {auroc_ci}",
                "test AUPRC": f"{t['auprc']:.3f} {auprc_ci}",
                "val AUROC": f"{r['val']['auroc']:.3f}",
                "val AUPRC": f"{r['val']['auprc']:.3f}",
                "n test": t["n"],
                "_sort": t["auprc"],
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("_sort", ascending=False).drop(columns="_sort") if len(df) else df


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def plot_curves(outputs_dir: Path, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    prevalence = None
    for pred_file in sorted(outputs_dir.glob("*/predictions.parquet")):
        d = pd.read_parquet(pred_file)
        d = d[d["split"] == "test"]
        name = pred_file.parent.name
        fpr, tpr, _ = roc_curve(d["label"], d["score"])
        prec, rec, _ = precision_recall_curve(d["label"], d["score"])
        ax1.plot(fpr, tpr, label=name)
        ax2.plot(rec, prec, label=name)
        prevalence = d["label"].mean()
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax1.set(
        title="ROC (test chromosomes)", xlabel="False positive rate", ylabel="True positive rate"
    )
    if prevalence is not None:
        ax2.axhline(prevalence, color="k", ls="--", lw=0.8, label="chance")
    ax2.set(title="Precision-recall (test chromosomes)", xlabel="Recall", ylabel="Precision")
    ax1.legend(loc="lower right")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()


@click.command()
@click.option("--results-dir", default="reports/results", show_default=True)
@click.option("--outputs-dir", default="outputs", show_default=True)
def main(results_dir: str, outputs_dir: str) -> None:
    table = results_table(Path(results_dir))
    md = to_markdown(table) if len(table) else "No results yet."
    Path(results_dir, "summary.md").write_text(md + "\n")
    click.echo(md)
    plot_curves(Path(outputs_dir), Path("reports/figures/04_roc_pr.png"))
    click.echo("\nSaved reports/results/summary.md and reports/figures/04_roc_pr.png")


if __name__ == "__main__":
    main()
