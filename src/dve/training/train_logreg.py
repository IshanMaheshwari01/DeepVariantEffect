"""Train and evaluate the logistic-regression baseline.

Usage:
    uv run python -m dve.training.train_logreg --config configs/logreg.yaml
"""

from __future__ import annotations

import time
from pathlib import Path

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression

from dve.eval.metrics import evaluate
from dve.models.baselines import logreg_features
from dve.utils import load_yaml, set_seed, write_json


@click.command()
@click.option("--config", "config_path", default="configs/logreg.yaml", show_default=True)
def main(config_path: str) -> None:
    cfg = load_yaml(config_path)
    set_seed(cfg["seed"])
    t0 = time.time()

    df = pd.read_parquet(cfg["data"])
    click.echo(f"Loaded {len(df):,} variants")
    X = logreg_features(df, cfg)
    y = df["label"].to_numpy()
    click.echo(f"Feature matrix: {X.shape[0]:,} x {X.shape[1]:,}")

    train = (df["split"] == "train").to_numpy()
    model = LogisticRegression(C=cfg["C"], max_iter=cfg["max_iter"], solver="lbfgs")
    model.fit(X[train], y[train])

    results = {"model": cfg["name"], "config": cfg}
    preds = []
    for split in ["val", "test"]:
        m = (df["split"] == split).to_numpy()
        score = model.predict_proba(X[m])[:, 1]
        results[split] = evaluate(y[m], score)
        preds.append(df.loc[m, ["variation_id", "gene", "label", "split"]].assign(score=score))
        r = results[split]
        click.echo(f"{split:>5}: AUROC {r['auroc']:.3f}  AUPRC {r['auprc']:.3f}  (n={r['n']:,})")

    results["train_seconds"] = round(time.time() - t0, 1)
    out_dir = Path("outputs") / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(preds).to_parquet(out_dir / "predictions.parquet", index=False)
    write_json(results, Path("reports/results") / f"{cfg['name']}.json")
    click.echo(f"Saved results to reports/results/{cfg['name']}.json")


if __name__ == "__main__":
    main()
