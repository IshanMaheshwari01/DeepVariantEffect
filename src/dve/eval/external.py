"""Fetch CADD, REVEL and AlphaMissense for our variants from MyVariant.info (dbNSFP).

    uv run python -m dve.eval.external

Results are cached, so re-running only asks for what is missing.
"""

from __future__ import annotations

import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import requests

URL = "https://myvariant.info/v1/variant"
FIELDS = "dbnsfp.cadd.phred,cadd.phred,dbnsfp.revel.score,dbnsfp.alphamissense.score"
OUT = Path("data/external/scores.parquet")
BATCH = 1000


def hgvs(chrom: str, pos: int, ref: str, alt: str) -> str:
    return f"chr{chrom}:g.{pos}{ref}>{alt}"


def _top(value) -> float:
    """Scores can be one number or a list with one value per transcript. Use the highest,
    the usual convention for REVEL and AlphaMissense when one variant has several."""
    if isinstance(value, dict):
        return _top(value.get("score", value.get("phred")))
    vals = value if isinstance(value, list) else [value]
    nums = [float(v) for v in vals if isinstance(v, (int, float))]
    return max(nums) if nums else np.nan


def parse_hit(hit: dict) -> dict:
    db = hit.get("dbnsfp") or {}
    if isinstance(db, list):  # rare: several dbNSFP records for one id
        db = db[0]
    cadd = (db.get("cadd") or {}).get("phred")
    if cadd is None:
        cadd = (hit.get("cadd") or {}).get("phred")
    return {
        "hgvs": hit.get("query", hit.get("_id")),
        "cadd_phred": _top(cadd),
        "revel": _top((db.get("revel") or {}).get("score")),
        "alphamissense": _top((db.get("alphamissense") or {}).get("score")),
    }


def fetch_batch(ids: list[str], session=requests, retries: int = 5) -> list[dict]:
    for attempt in range(retries):
        try:
            r = session.post(
                URL,
                params={"assembly": "hg38"},
                data={"ids": ",".join(ids), "fields": FIELDS},
                timeout=120,
            )
            r.raise_for_status()
            return [parse_hit(h) for h in r.json() if not h.get("notfound")]
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    return []


@click.command()
@click.option("--data", default="data/processed/variants.parquet", show_default=True)
@click.option("--min-coverage", default=0.5, show_default=True)
def main(data, min_coverage):
    cols = ["chrom", "pos", "ref", "alt", "split"]
    df = pd.read_parquet(data, columns=cols)
    df = df[df["split"].isin(["val", "test"])].copy()
    df["hgvs"] = [hgvs(*t) for t in zip(df["chrom"], df["pos"], df["ref"], df["alt"], strict=True)]
    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["hgvs"])
    todo = sorted(set(df["hgvs"]) - set(done["hgvs"]))
    rows = []
    for i in range(0, len(todo), BATCH):
        rows += fetch_batch(todo[i : i + BATCH])
        print(f"{min(i + BATCH, len(todo)):,}/{len(todo):,}", flush=True)
        time.sleep(0.3)
    frames = [f for f in (done, pd.DataFrame(rows)) if len(f)]
    scores = pd.concat(frames, ignore_index=True).drop_duplicates("hgvs")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(OUT, index=False)
    m = df.merge(scores, on="hgvs", how="left")
    cover = m[["cadd_phred", "revel", "alphamissense"]].notna().mean()
    print("coverage:\n", cover.round(3).to_string())
    if cover.min() < min_coverage:
        raise SystemExit("Coverage is too low: check the MyVariant fields before benchmarking.")


if __name__ == "__main__":
    main()
