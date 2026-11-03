"""Train and evaluate the 1D-CNN baseline.

Usage:
    uv run python -m dve.training.train_cnn --config configs/cnn.yaml
"""

from __future__ import annotations

import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from torch import nn

from dve.eval.metrics import evaluate
from dve.features import center_crop, encode, one_hot_pair
from dve.models.cnn import VariantCNN, reverse_complement
from dve.utils import load_yaml, set_seed, write_json


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_inputs(df: pd.DataFrame, flank_in: int, flank: int) -> torch.Tensor:
    ref = encode(center_crop(df["ref_seq"].tolist(), flank_in, flank))
    alt = encode(center_crop(df["alt_seq"].tolist(), flank_in, flank))
    return torch.from_numpy(one_hot_pair(ref, alt))


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, device, batch_size: int = 1024) -> np.ndarray:
    """Average predictions over forward and reverse-complement strands."""
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = X[i : i + batch_size].to(device)
        logits = (model(xb) + model(reverse_complement(xb))) / 2
        out.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(out)


@click.command()
@click.option("--config", "config_path", default="configs/cnn.yaml", show_default=True)
@click.option(
    "--max-train", default=0, help="Subsample training set (0 = use all). For quick tests."
)
def main(config_path: str, max_train: int) -> None:
    cfg = load_yaml(config_path)
    set_seed(cfg["seed"])
    device = pick_device()
    click.echo(f"Device: {device}")
    t0 = time.time()

    df = pd.read_parquet(cfg["data"])
    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ["train", "val", "test"]}
    if max_train:
        parts["train"] = parts["train"].sample(
            n=min(max_train, len(parts["train"])), random_state=cfg["seed"]
        )
    X = {s: make_inputs(d, cfg["flank_in"], cfg["flank"]) for s, d in parts.items()}
    y = {s: torch.tensor(d["label"].to_numpy(), dtype=torch.float32) for s, d in parts.items()}
    click.echo("Sizes: " + ", ".join(f"{s}={len(d):,}" for s, d in parts.items()))

    model = VariantCNN(channels=cfg["channels"], dropout=cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    loss_fn = nn.BCEWithLogitsLoss()

    out_dir = Path("outputs") / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    best_auprc, best_epoch, bad_epochs = -1.0, 0, 0
    n_train = len(X["train"])

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(n_train)
        total = 0.0
        for i in range(0, n_train, cfg["batch_size"]):
            idx = perm[i : i + cfg["batch_size"]]
            xb, yb = X["train"][idx].to(device), y["train"][idx].to(device)
            if cfg["rc_augment"] and torch.rand(1).item() < 0.5:
                xb = reverse_complement(xb)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()

        val = evaluate(y["val"].numpy(), predict(model, X["val"], device), n_boot=0)
        click.echo(
            f"epoch {epoch:2d}  loss {total / n_train:.4f}  "
            f"val AUROC {val['auroc']:.3f}  val AUPRC {val['auprc']:.3f}"
        )
        if val["auprc"] > best_auprc:
            best_auprc, best_epoch, bad_epochs = val["auprc"], epoch, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                click.echo("Early stopping.")
                break

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    results = {"model": cfg["name"], "config": cfg, "best_epoch": best_epoch}
    preds = []
    for split in ["val", "test"]:
        score = predict(model, X[split], device)
        results[split] = evaluate(y[split].numpy(), score)
        d = parts[split]
        preds.append(d[["variation_id", "gene", "label", "split"]].assign(score=score))
        r = results[split]
        click.echo(f"{split:>5}: AUROC {r['auroc']:.3f}  AUPRC {r['auprc']:.3f}  (n={r['n']:,})")

    results["train_seconds"] = round(time.time() - t0, 1)
    results["device"] = str(device)
    pd.concat(preds).to_parquet(out_dir / "predictions.parquet", index=False)
    write_json(results, Path("reports/results") / f"{cfg['name']}.json")
    click.echo(f"Saved results to reports/results/{cfg['name']}.json")


if __name__ == "__main__":
    main()
