"""Fine-tune Nucleotide Transformer with LoRA.

    uv run python -m dve.training.train_nt                 # GPU
    uv run python -m dve.training.train_nt --lora-r 0      # head only (cheaper)

Saves the best weights and a resume checkpoint to --out-dir, so a run stopped by a Colab
disconnect continues from the last validation check.
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from dve.eval.metrics import evaluate
from dve.features import encode
from dve.models import nt
from dve.models.nt import revcomp
from dve.training.loop import EarlyStopper, device, pos_weight
from dve.utils import set_seed, write_json


def check_tokenizer(seqs: list[str]) -> None:
    """Stop before training if our fast tokenizer disagrees with the real one."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(nt.MODEL_NAME)
    theirs = np.array(tok(seqs, add_special_tokens=True)["input_ids"])
    ours = nt.tokenize(encode(seqs))
    if not np.array_equal(theirs, ours):
        raise RuntimeError("Tokenizer mismatch: check nt.tokenize against the model vocab")


@torch.no_grad()
def predict(model, toks, dev, bs) -> np.ndarray:
    """toks = (ref, alt, ref_rc, alt_rc). Returns logits averaged over both strands."""
    model.eval()
    out = []
    amp = dev.type == "cuda"
    for i in tqdm(range(0, len(toks[0]), bs), desc="predict", mininterval=30):
        t = [torch.from_numpy(x[i : i + bs]).to(dev) for x in toks]
        with torch.autocast(dev.type, enabled=amp):
            logit = (model(t[0], t[1]) + model(t[2], t[3])) / 2
        out.append(logit.float().cpu().numpy())
    return np.concatenate(out)


@click.command()
@click.option("--data", default="data/processed/variants.parquet", show_default=True)
@click.option("--ckpt-dir", default="outputs/nt_ckpt", show_default=True)
@click.option("--max-train", default=60000, show_default=True, help="Stratified train subset.")
@click.option("--epochs", default=3, show_default=True)
@click.option("--batch-size", default=8, show_default=True, help="Lower this if out of memory.")
@click.option("--grad-accum", default=4, show_default=True)
@click.option("--eval-every", default=2000, show_default=True, help="Optimiser steps.")
@click.option("--lr-lora", default=1e-4, show_default=True)
@click.option("--lr-head", default=1e-3, show_default=True)
@click.option("--lora-r", default=8, show_default=True, help="0 = train the head only.")
@click.option("--patience", default=4, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--tiny", is_flag=True, help="Random tiny model, for testing code only.")
def main(
    data,
    ckpt_dir,
    max_train,
    epochs,
    batch_size,
    grad_accum,
    eval_every,
    lr_lora,
    lr_head,
    lora_r,
    patience,
    seed,
    tiny,
):
    set_seed(seed)
    dev = device()
    df = pd.read_parquet(data)
    train = df[df["split"] == "train"]
    if max_train < len(train):
        keep = train.groupby("label").sample(frac=max_train / len(train), random_state=seed)
        df = pd.concat([keep, df[df["split"] != "train"]], ignore_index=True)
    if not tiny:
        check_tokenizer(df["ref_seq"].head(4).tolist())
    r, a = encode(df["ref_seq"].tolist()), encode(df["alt_seq"].tolist())
    toks = tuple(nt.tokenize(x) for x in (r, a, revcomp(r), revcomp(a)))
    del r, a
    y = df["label"].values.astype(np.float32)
    tr, va = np.flatnonzero(df["split"] == "train"), np.flatnonzero(df["split"] == "val")

    model = nt.build(tiny=tiny, lora_r=lora_r, window=len(df["ref_seq"].iloc[0])).to(dev)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    opt = torch.optim.AdamW(
        [{"params": lora, "lr": lr_lora}, {"params": model.head.parameters(), "lr": lr_head}],
        weight_decay=0.01,
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"device={dev} train={len(tr):,} trainable params={n_train:,}")
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight(y[tr], dev))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    stopper = EarlyStopper(patience)

    out = Path(ckpt_dir)
    out.mkdir(parents=True, exist_ok=True)
    start_epoch, start_batch, step = 0, 0, 0
    if (out / "resume.pt").exists():
        st = torch.load(out / "resume.pt", map_location=dev, weights_only=False)
        model.load_state_dict(st["model"], strict=False)
        opt.load_state_dict(st["opt"])
        start_epoch, start_batch, step = st["epoch"], st["batch"], st["step"]
        stopper.best, stopper.bad = st["best"], st["bad"]
        print(f"resumed at epoch {start_epoch}, batch {start_batch}")

    def validate(epoch, batch) -> bool:
        ap = average_precision_score(y[va], predict(model, [t[va] for t in toks], dev, batch_size))
        improved, stop = stopper.step(ap)
        print(f"step {step}: val AUPRC {ap:.4f}{' *' if improved else ''}", flush=True)
        state = nt.trainable_state(model)
        if improved:
            torch.save(state, out / "best.pt")
        torch.save(
            {
                "model": state,
                "opt": opt.state_dict(),
                "epoch": epoch,
                "batch": batch,
                "step": step,
                "best": stopper.best,
                "bad": stopper.bad,
            },
            out / "resume.pt",
        )
        model.train()
        return stop

    stop = False
    for epoch in range(start_epoch, epochs):
        # the permutation depends only on seed + epoch, so a resumed run sees the same order
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(tr)
        flips = rng.random(len(order) // batch_size + 1) < 0.5
        model.train()
        first = start_batch if epoch == start_epoch else 0
        bar = tqdm(
            range(first, len(order) // batch_size + 1), desc=f"epoch {epoch}", mininterval=30
        )
        for bi in bar:
            b = order[bi * batch_size : (bi + 1) * batch_size]
            if len(b) == 0:
                continue
            pair = (2, 3) if flips[bi] else (0, 1)
            ref, alt = (torch.from_numpy(toks[j][b]).to(dev) for j in pair)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = loss_fn(model(ref, alt), torch.from_numpy(y[b]).to(dev)) / grad_accum
            scaler.scale(loss).backward()
            if (bi + 1) % grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % eval_every == 0 and validate(epoch, bi + 1):
                    stop = True
                    break
        if stop:
            break
        if validate(epoch + 1, 0):
            break

    model.load_state_dict(torch.load(out / "best.pt", map_location=dev), strict=False)
    name = "nt_lora" if lora_r else "nt_frozen"
    result = {
        "model": name,
        "smoke": tiny,
        "config": {
            "n_train": len(tr),
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "lr_lora": lr_lora,
            "lr_head": lr_head,
            "lora_r": lora_r,
            "seed": seed,
        },
    }
    preds = []
    for split in ["val", "test"]:
        m = np.flatnonzero(df["split"] == split)
        score = expit(predict(model, [t[m] for t in toks], dev, batch_size))
        result[split] = evaluate(y[m], score)
        d = df.iloc[m][["variation_id", "gene", "label", "split"]]
        preds.append(d.assign(score=score))
        r = result[split]
        click.echo(f"{split:>5}: AUROC {r['auroc']:.3f}  AUPRC {r['auprc']:.3f}  (n={r['n']:,})")
    pred_dir = Path("outputs") / name
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(preds).to_parquet(pred_dir / "predictions.parquet", index=False)
    write_json(result, Path("reports/results") / f"{name}.json")


if __name__ == "__main__":
    main()
