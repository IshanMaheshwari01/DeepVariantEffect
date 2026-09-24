"""Quick exploratory plots of the processed variant dataset.

Usage:
    uv run python scripts/eda_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"benign": "#4C72B0", "pathogenic": "#C44E52"}


def _labelled(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(cls=df["label"].map({0: "benign", 1: "pathogenic"}))


def plot_split_balance(df: pd.DataFrame, out: Path) -> None:
    counts = _labelled(df).groupby(["split", "cls"]).size().unstack(fill_value=0)
    counts = counts.reindex(["train", "val", "test"]).dropna(how="all")
    ax = counts.plot(kind="bar", color=[COLORS[c] for c in counts.columns], figsize=(6, 4))
    ax.set_title("Variants per split")
    ax.set_xlabel("")
    ax.set_ylabel("Number of variants")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_top_genes(df: pd.DataFrame, out: Path, n: int = 20) -> None:
    d = _labelled(df)
    top = d["gene"].value_counts().head(n).index
    counts = d[d["gene"].isin(top)].groupby(["gene", "cls"]).size().unstack(fill_value=0)
    counts = counts.loc[counts.sum(axis=1).sort_values().index]
    ax = counts.plot(
        kind="barh", stacked=True, color=[COLORS[c] for c in counts.columns], figsize=(7, 7)
    )
    ax.set_title(f"Top {n} genes by labelled missense variants")
    ax.set_xlabel("Number of variants")
    ax.set_ylabel("")
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_review_stars(df: pd.DataFrame, out: Path) -> None:
    counts = _labelled(df).groupby(["stars", "cls"]).size().unstack(fill_value=0)
    ax = counts.plot(kind="bar", color=[COLORS[c] for c in counts.columns], figsize=(6, 4))
    ax.set_title("ClinVar review status (gold stars)")
    ax.set_xlabel("Stars")
    ax.set_ylabel("Number of variants")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


@click.command()
@click.option("--parquet", default="data/processed/variants.parquet", show_default=True)
@click.option("--outdir", default="reports/figures", show_default=True)
def main(parquet: str, outdir: str) -> None:
    df = pd.read_parquet(parquet, columns=["gene", "label", "split", "stars"])
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    plot_split_balance(df, out / "01_split_balance.png")
    plot_top_genes(df, out / "02_top_genes.png")
    plot_review_stars(df, out / "03_review_stars.png")
    click.echo(f"Saved 3 figures to {out}/")


if __name__ == "__main__":
    main()
