"""Prepare the small files the demo needs. Only held-out test variants are exported.

uv run python scripts/export_app_data.py
"""

import json
from pathlib import Path

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression

from dve.eval.stats import _logit


@click.command()
@click.option("--data", default="data/processed/variants.parquet", show_default=True)
def main(data):
    p = pd.read_parquet("outputs/cnn_onehot/predictions.parquet")
    va = p[p["split"] == "val"]
    lr = LogisticRegression(C=1e6).fit(_logit(va["score"].values).reshape(-1, 1), va["label"])
    Path("models").mkdir(exist_ok=True)
    Path("models/app_meta.json").write_text(
        json.dumps({"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])})
    )
    df = pd.read_parquet(data)
    cols = [
        "variation_id",
        "chrom",
        "pos",
        "ref",
        "alt",
        "gene",
        "clnsig",
        "stars",
        "ref_seq",
        "alt_seq",
    ]
    df[df["split"] == "test"][cols].to_parquet(
        "reports/app_variants.parquet", index=False, compression="zstd"
    )
    print("wrote models/app_meta.json and reports/app_variants.parquet")


if __name__ == "__main__":
    main()
