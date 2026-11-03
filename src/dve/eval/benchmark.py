"""Compare every model with CADD, REVEL and AlphaMissense on the same test variants.

uv run python -m dve.eval.benchmark
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from dve.eval import stats as M  # noqa: E402
from dve.eval.external import OUT as SCORES  # noqa: E402
from dve.eval.external import hgvs  # noqa: E402

OURS = {
    "logreg_kmer": "Logistic regression",
    "cnn_onehot": "1D CNN",
    "nt_frozen": "NT, frozen",
    "nt_lora": "NT + LoRA",
}
THEIRS = {"cadd_phred": "CADD", "revel": "REVEL", "alphamissense": "AlphaMissense"}
LABELS = {**OURS, **THEIRS}
# fixed colour per model, so one model is the same colour in every figure
COLORS = dict(
    zip(
        LABELS,
        ["#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
        strict=True,
    )
)
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
    }
)


def load(data="data/processed/variants.parquet", allow_test=False):
    """One table of all model scores and external scores.

    Refuses a mock dataset or a smoke run unless allow_test is set."""
    cols = ["variation_id", "chrom", "pos", "ref", "alt", "gene", "stars", "label", "split"]
    all_cols = pq.ParquetFile(data).schema_arrow.names
    if "mock" in all_cols and not allow_test:
        raise SystemExit(f"{data} is a mock dataset: use the real one.")
    base = pd.read_parquet(data, columns=cols)
    base = base[base["split"].isin(["val", "test"])].copy()
    base["variation_id"] = base["variation_id"].astype(str)
    models = []
    for name in OURS:
        path, res = (
            Path("outputs") / name / "predictions.parquet",
            Path("reports/results") / f"{name}.json",
        )
        if not path.exists():
            continue
        if json.loads(res.read_text()).get("smoke") and not allow_test:
            raise SystemExit(f"{name} is a smoke run: retrain it on the real data.")
        p = pd.read_parquet(path, columns=["variation_id", "score"])
        p["variation_id"] = p["variation_id"].astype(str)
        base = base.merge(p.rename(columns={"score": name}), on="variation_id", how="inner")
        models.append(name)
    if not models:
        raise SystemExit("No predictions found in outputs/. Train a model first.")
    base["hgvs"] = [
        hgvs(*t) for t in zip(base["chrom"], base["pos"], base["ref"], base["alt"], strict=True)
    ]
    ext = (
        pd.read_parquet(SCORES)
        if Path(SCORES).exists()
        else pd.DataFrame(columns=["hgvs", *THEIRS])
    )
    return base.merge(ext, on="hgvs", how="left"), models


def analyse(df: pd.DataFrame, models: list[str]) -> dict:
    te, va = df[df["split"] == "test"], df[df["split"] == "val"]
    tools = [t for t in THEIRS if te[t].notna().any()]
    common = te.dropna(subset=tools)
    res = {
        "n_test": len(te),
        "n_common": len(common),
        "models": models,
        "tools": tools,
        "common": {},
        "vs_revel": {},
        "by_stars": {},
    }
    y, g = common["label"].values, common["gene"].values
    for name in [*models, *tools]:
        s = common[name].values
        res["common"][name] = {
            "auroc": M.auroc(y, s),
            "auprc": M.auprc(y, s),
            "auroc_ci": M.bootstrap_ci(y, s, g, M.auroc),
            "auprc_ci": M.bootstrap_ci(y, s, g, M.auprc),
            "within_gene_auroc": M.within_gene_auroc(y, s, g)[0],
        }
        if "revel" in tools and name != "revel":
            d, lo, hi = M.paired_delta_ci(y, s, common["revel"].values, g)
            res["vs_revel"][name] = {"delta_auroc": d, "ci": [lo, hi]}
    for label, mask in (
        ("1 star", common["stars"] == 1),
        ("2 stars", common["stars"] == 2),
        ("3+ stars", common["stars"] >= 3),
    ):
        sub = common[mask]
        if sub["label"].nunique() == 2 and sub["label"].sum() >= 20:
            res["by_stars"][label] = {
                "n": len(sub),
                "pathogenic": int(sub["label"].sum()),
                **{n: M.auroc(sub["label"], sub[n]) for n in [*models, *tools]},
            }
    res["calibration"] = {
        n: {"ece": M.ece(te["label"], M.platt_calibrate(va["label"], va[n], te[n]))} for n in models
    }
    return res


def fig_curves(df, names, out):
    te = df[df["split"] == "test"].dropna(subset=list(names))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 5.2), constrained_layout=True)
    y = te["label"].values
    for n in names:
        x1, y1, _ = roc_curve(y, te[n])
        a.plot(x1, y1, color=COLORS[n], lw=2, label=LABELS[n])
        p, r, _ = precision_recall_curve(y, te[n])
        b.plot(r, p, color=COLORS[n], lw=2)
    a.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
    b.axhline(y.mean(), color=MUTED, lw=1, ls=":")
    a.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    b.set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    fig.legend(frameon=False, fontsize=9, loc="outside lower center", ncol=4)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_reliability(df, models, out):
    te, va = df[df["split"] == "test"], df[df["split"] == "val"]
    fig, ax = plt.subplots(figsize=(5.2, 4.8), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
    for n in models:
        p = M.platt_calibrate(va["label"], va[n], te[n])
        x, y = M.reliability(te["label"], p)
        ax.plot(x, y, color=COLORS[n], lw=2, marker="o", ms=4, label=LABELS[n])
    ax.set(
        xlabel="Predicted probability (after Platt scaling)",
        ylabel="Observed pathogenic",
        title="Calibration on test variants",
    )
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_stars(res, out):
    strata = list(res["by_stars"])
    names = [*res["models"], *res["tools"]]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    for n in names:
        vals = [res["by_stars"][s][n] for s in strata]
        ax.plot(strata, vals, color=COLORS[n], lw=2, marker="o", ms=8, label=LABELS[n])
    ax.set_xticks(range(len(strata)), [f"{s}\n(n={res['by_stars'][s]['n']:,})" for s in strata])
    ax.grid(axis="y", color=MUTED, alpha=0.15)
    ax.set(ylabel="AUROC", title="AUROC by ClinVar review status")
    fig.legend(frameon=False, fontsize=9, loc="outside lower center", ncol=4)
    fig.savefig(out, dpi=200)
    plt.close(fig)


@click.command()
@click.option("--data", default="data/processed/variants.parquet", show_default=True)
@click.option("--allow-test", is_flag=True, help="Accept mock runs, for testing the code only.")
@click.option("--out-dir", default="reports", show_default=True)
def main(data, allow_test, out_dir):
    df, models = load(data, allow_test)
    res = analyse(df, models)
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    fig_curves(df, [*models, *res["tools"]], out / "figures/04_roc_pr.png")
    fig_reliability(df, models, out / "figures/05_calibration.png")
    if res["by_stars"]:
        fig_stars(res, out / "figures/06_by_review_status.png")
    (out / "results/benchmark.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res["common"], indent=2))


if __name__ == "__main__":
    main()
