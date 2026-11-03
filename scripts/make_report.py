"""Fill the README results table and write reports/REPORT.md from saved results.

Every number comes from reports/results/*.json. Mock and smoke runs are refused, so test
output can never reach the README.

    uv run python scripts/make_report.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import click

from dve.eval.benchmark import LABELS

RES = Path("reports/results")
START, END = "<!-- results:start -->", "<!-- results:end -->"


def load_real(allow_test: bool) -> tuple[dict, dict]:
    bench = json.loads((RES / "benchmark.json").read_text())
    runs = {}
    for name in bench["models"]:
        run = json.loads((RES / f"{name}.json").read_text())
        if run.get("smoke") and not allow_test:
            raise SystemExit(f"Refusing to report {name}: it is a smoke run.")
        runs[name] = run
    return bench, runs


def table(bench: dict) -> str:
    rows = [
        "| Model | AUROC (95% CI) | AUPRC (95% CI) | Within-gene AUROC |",
        "|---|:---:|:---:|:---:|",
    ]
    order = sorted(bench["common"], key=lambda n: -bench["common"][n]["auroc"])
    for n in order:
        c = bench["common"][n]
        rows.append(
            f"| {LABELS[n]} | {c['auroc']:.3f} ({c['auroc_ci'][0]:.3f}-{c['auroc_ci'][1]:.3f}) "
            f"| {c['auprc']:.3f} ({c['auprc_ci'][0]:.3f}-{c['auprc_ci'][1]:.3f}) "
            f"| {c['within_gene_auroc']:.3f} |"
        )
    return "\n".join(rows)


def findings(bench: dict) -> list[str]:
    """Plain statements computed from the results, in place of conclusions written ahead."""
    c, out = bench["common"], []
    seq = bench["models"]
    best = max(seq, key=lambda n: c[n]["auroc"])
    out.append(f"The best sequence-only model is {LABELS[best]}, AUROC {c[best]['auroc']:.3f}.")
    if "nt_lora" in seq and "cnn_onehot" in seq:
        out.append(
            "The transformer changes AUROC by "
            f"{c['nt_lora']['auroc'] - c['cnn_onehot']['auroc']:+.3f} compared with the CNN."
        )
    if best in bench.get("vs_revel", {}):
        d = bench["vs_revel"][best]
        out.append(
            f"Against REVEL the difference is {d['delta_auroc']:+.3f} "
            f"(95% CI {d['ci'][0]:+.3f} to {d['ci'][1]:+.3f}, resampling genes)."
        )
    return out


def stars_table(bench: dict) -> str:
    if not bench["by_stars"]:
        return "Too few pathogenic variants in each review tier to report."
    names = [*bench["models"], *bench["tools"]]
    head = "| Review status | n | Pathogenic | " + " | ".join(LABELS[n] for n in names) + " |"
    rule = "|---|---:|---:|" + ":---:|" * len(names)
    rows = [
        f"| {s} | {v['n']:,} | {v['pathogenic']:,} | "
        + " | ".join(f"{v[n]:.3f}" for n in names)
        + " |"
        for s, v in bench["by_stars"].items()
    ]
    return "\n".join([head, rule, *rows])


def report(bench: dict, runs: dict) -> str:
    ece = "\n".join(f"- {LABELS[n]}: ECE {v['ece']:.3f}" for n, v in bench["calibration"].items())
    params = "\n".join(
        f"- {LABELS[n]}: `{json.dumps(r.get('config', {}))}`" for n, r in runs.items()
    )
    return f"""# DeepVariantEffect: Report

## Question

How well can a model separate pathogenic from benign missense variants using only the DNA
around them, compared with established tools that use conservation and protein features?

## Data

ClinVar missense SNVs with at least one review star, 1,025 bp windows from GRCh38, split by
chromosome: test on chr8 and chr21, validate on chr10, train on the rest. The split keeps
near-identical windows apart, but genes on test chromosomes can still have paralogs in training.

## Methods

Windows are on the genome's forward strand, while genes can lie on either strand, so all models
use strand handling: substitution classes are collapsed to one strand for logistic regression,
and the neural models train with reverse-complement augmentation and average both strands.

- **Logistic regression:** 96 trinucleotide substitution classes, context 4-mer frequencies
  and the 6-mers the variant creates or destroys.
- **1D CNN:** three convolutional blocks on one-hot ref and alt, over 201 bp.
- **NT + LoRA:** Nucleotide Transformer (24 layers, frozen) with trainable LoRA adapters on
  the query and value projections of each attention layer. The head reads the token holding
  the variant and the mean of all tokens, for ref and for alt minus ref.

Model selection used validation AUPRC only. Test variants were used once, for the numbers below.
CIs come from 500 bootstrap resamples of genes, because variants in one gene are correlated.
Probabilities were calibrated by Platt scaling on the validation set.

## Results

On the {bench["n_common"]:,} of {bench["n_test"]:,} test variants that have a score from every
external tool, so all rows use identical variants.

{table(bench)}

{chr(10).join("- " + f for f in findings(bench))}

![ROC and PR curves](figures/04_roc_pr.png)

### By review status

{stars_table(bench)}

![AUROC by review status](figures/06_by_review_status.png)

### Calibration

{ece}

![Calibration](figures/05_calibration.png)

## Limitations

- CADD, REVEL and AlphaMissense may have been trained or tuned on variants that overlap this
  test set, so their scores here can be optimistic. This model's chromosome split is stricter.
- ClinVar labels are biased toward well-studied genes and European ancestry.
- Some "benign" labels partly reflect allele frequency, so a test on ClinVar is not a test
  on the variants of uncertain significance the tool would actually be used for.
- Missense annotation is by any transcript, so some pathogenic variants may act through
  splicing rather than the amino acid change.
- Not a clinical tool.

## Run details

{params}
"""


def update_readme(bench: dict, path: Path) -> None:
    text = path.read_text()
    block = (
        f"{START}\nOn {bench['n_common']:,} test variants scored by every tool:\n\n"
        f"{table(bench)}\n\n"
        + "\n".join(f"- {f}" for f in findings(bench))
        + f"\n\nFull write-up: [`reports/REPORT.md`](reports/REPORT.md).\n{END}"
    )
    new = re.sub(re.escape(START) + ".*" + re.escape(END), lambda _: block, text, flags=re.S)
    if new == text and block not in text:
        raise SystemExit("Results markers not found in README.md")
    path.write_text(new)


@click.command()
@click.option("--allow-test", is_flag=True, help="Test the code on mock runs.")
@click.option("--out-dir", default="reports", show_default=True)
@click.option("--readme", default="README.md", show_default=True)
def main(allow_test, out_dir, readme):
    bench, runs = load_real(allow_test)
    Path(out_dir, "REPORT.md").write_text(report(bench, runs))
    update_readme(bench, Path(readme))
    print("Updated", readme, "and", Path(out_dir, "REPORT.md"))


if __name__ == "__main__":
    main()
