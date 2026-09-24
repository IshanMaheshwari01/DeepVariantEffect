"""Build the labelled variant dataset: ClinVar + GRCh38 -> Parquet.

Usage:
    uv run python -m dve.data.build_dataset --config configs/data.yaml
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd
import yaml
from tqdm import tqdm

from dve.data.clinvar import clinvar_file_date, iter_clinvar
from dve.data.reference import Reference, make_alt


def assign_split(chrom: str, splits: dict[str, list[str]]) -> str:
    for name, chroms in splits.items():
        if chrom in chroms:
            return name
    return "train"


def build(cfg: dict) -> tuple[pd.DataFrame, dict]:
    flank = int(cfg["flank"])
    ref = Reference(cfg["reference_fasta"])

    rows = []
    dropped = {"ref_mismatch": 0, "off_chromosome": 0, "contains_N": 0, "duplicate": 0}
    seen: set[tuple[str, int, str, str]] = set()

    records = iter_clinvar(cfg["clinvar_vcf"], cfg["chromosomes"], cfg["min_review_stars"])
    for r in tqdm(records, desc="ClinVar missense SNVs", unit=" var"):
        key = (r.chrom, r.pos, r.ref, r.alt)
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)

        ref_seq = ref.window(r.chrom, r.pos, flank)
        if ref_seq is None:
            dropped["off_chromosome"] += 1
            continue
        if ref_seq[flank] != r.ref:
            dropped["ref_mismatch"] += 1
            continue
        if "N" in ref_seq:
            dropped["contains_N"] += 1
            continue

        rows.append(
            {
                "variation_id": r.variation_id,
                "chrom": r.chrom,
                "pos": r.pos,
                "ref": r.ref,
                "alt": r.alt,
                "gene": r.gene,
                "clnsig": r.clnsig,
                "review_status": r.review_status,
                "stars": r.stars,
                "label": r.label,
                "split": assign_split(r.chrom, cfg["splits"]),
                "ref_seq": ref_seq,
                "alt_seq": make_alt(ref_seq, flank, r.alt),
            }
        )

    df = pd.DataFrame(rows)
    summary = {
        "clinvar_release": clinvar_file_date(cfg["clinvar_vcf"]),
        "flank": flank,
        "window_length": 2 * flank + 1,
        "min_review_stars": cfg["min_review_stars"],
        "n_variants": int(len(df)),
        "n_genes": int(df["gene"].nunique()) if len(df) else 0,
        "dropped": dropped,
        "by_split": (
            df.groupby(["split", "label"])
            .size()
            .unstack(fill_value=0)
            .rename(columns={0: "benign", 1: "pathogenic"})
            .to_dict(orient="index")
            if len(df)
            else {}
        ),
    }
    return df, summary


@click.command()
@click.option("--config", "config_path", default="configs/data.yaml", show_default=True)
def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    df, summary = build(cfg)

    out = Path(cfg["output_parquet"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, compression="zstd")

    summary_path = Path(cfg["summary_json"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    click.echo(f"\nWrote {len(df):,} variants -> {out}")
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
