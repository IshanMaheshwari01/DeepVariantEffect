"""Write a small fake dataset for testing the code path. Its results mean nothing.

The `mock` column makes every result carry mock=true, which the report step rejects.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed/mock.parquet")
rng = np.random.default_rng(0)
base = np.array(list("ACGT"))
rows = []
for i in range(n):
    s = rng.choice(base, 1025)
    label = int(rng.random() < 0.3)
    s[512] = "C" if label else "A"  # planted signal
    alt = "T" if label else "G"
    chrom, split = [("1", "train"), ("10", "val"), ("8", "test")][
        rng.choice(3, p=[0.7, 0.15, 0.15])
    ]
    ref_seq = "".join(s)
    rows.append(
        dict(
            variation_id=str(i),
            chrom=chrom,
            pos=1000 + i,
            ref=s[512],
            alt=alt,
            gene=f"G{i % 60}",
            clnsig="",
            review_status="",
            stars=1 + i % 3,
            label=label,
            split=split,
            ref_seq=ref_seq,
            alt_seq=ref_seq[:512] + alt + ref_seq[513:],
            mock=True,
        )
    )
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_parquet(out, index=False)
print(f"wrote {out}")
