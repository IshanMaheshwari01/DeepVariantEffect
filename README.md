# DeepVariantEffect

Predicting whether a missense variant is pathogenic or benign from the DNA sequence around it.

**Status: code complete, full training run pending.** The dataset is built. All models, the benchmark and the demo are implemented and tested. The results table fills in from the real run.

## The idea

After variant calling (see my [gatk4-germline-pipeline](https://github.com/IshanMaheshwari01/gatk4-germline-pipeline)) you end up with a VCF full of variants, and most of them have never been classified. In ClinVar they would be "variants of uncertain significance". Tools like CADD and REVEL give a score for these, but they combine a lot of hand-made features.

I want to see how far you can get using only the DNA sequence. The plan is to fine-tune a DNA language model (Nucleotide Transformer from InstaDeep) on ClinVar variants and compare it against CADD and REVEL on the same held-out variants.

## Plan

```
ClinVar VCF  ->  keep missense SNVs with a clear label (pathogenic = 1, benign = 0)
GRCh38 FASTA ->  take 512 bp on each side of the variant (1,025 bp window)
             ->  build the reference and the alternate sequence
             ->  Nucleotide Transformer (LoRA fine-tuning) on both
             ->  difference between the two embeddings
             ->  small classifier  ->  P(pathogenic)
```

Before the transformer I train two simple baselines, logistic regression on substitution and k-mer features and a 1D CNN on one-hot sequence, so I know whether the big model is actually adding anything.

## Dataset (done)

Built from the ClinVar release of 13 September 2026 and the Ensembl GRCh38 primary assembly. I kept missense SNVs labelled pathogenic / likely pathogenic or benign / likely benign, with at least one review star, and dropped conflicting ones.

To avoid leakage, I split by chromosome instead of at random. Variants close together on the same chromosome can share almost the same sequence window, so a random split would let the model see near copies of the test set.

| Split | Chromosomes | Benign | Pathogenic |
|---|---|---:|---:|
| Train | everything else | 134,144 | 57,619 |
| Validation | chr10 | 4,894 | 1,684 |
| Test | chr8, chr21 | 6,976 | 1,701 |

That's 207,018 variants across 15,018 genes. Every sequence window was checked against the reference base, and none had a mismatch or an N. The full summary is in `reports/data_summary.json` and the EDA plots are in `reports/figures/`.

The classes are imbalanced (about 3 benign to 1 pathogenic), so I'll report AUPRC as well as AUROC.

## Results

<!-- results:start -->
On 8,168 test variants scored by every tool:

| Model | AUROC (95% CI) | AUPRC (95% CI) | Within-gene AUROC |
|---|:---:|:---:|:---:|
| REVEL | 0.973 (0.965-0.979) | 0.921 (0.885-0.948) | 0.953 |
| AlphaMissense | 0.954 (0.941-0.967) | 0.851 (0.789-0.903) | 0.959 |
| CADD | 0.926 (0.909-0.945) | 0.723 (0.634-0.809) | 0.912 |
| 1D CNN | 0.775 (0.745-0.804) | 0.497 (0.388-0.591) | 0.753 |
| Logistic regression | 0.701 (0.678-0.727) | 0.352 (0.277-0.440) | 0.676 |

- The best sequence-only model is 1D CNN, AUROC 0.775.
- Against REVEL the difference is -0.198 (95% CI -0.229 to -0.168, resampling genes).

Full write-up: [`reports/REPORT.md`](reports/REPORT.md).
<!-- results:end -->

## Running it

```bash
# install uv first: https://docs.astral.sh/uv/
git clone https://github.com/IshanMaheshwari01/DeepVariantEffect.git
cd DeepVariantEffect
uv sync --all-extras

uv run pytest -q                  # tests
bash scripts/download_data.sh     # ClinVar VCF (~190 MB) and GRCh38 FASTA (~900 MB)
uv run python -m dve.data.build_dataset
uv run python scripts/eda_dataset.py

bash scripts/run_all.sh                       # baselines, CNN, benchmark, report
uv run python -m dve.training.train_nt        # transformer (needs a GPU)
uv run streamlit run src/dve/app/streamlit_app.py
```

To train the transformer on Colab use `notebooks/train_nt_colab.ipynb`. After it finishes, re-run the last four commands in `scripts/run_all.sh`.

Settings for the dataset (flank size, review stars, splits) are in `configs/data.yaml`. Lint and tests run on every push with GitHub Actions.

## Roadmap

- [x] Repo setup, environment, CI
- [x] ClinVar + GRCh38 dataset with a chromosome split
- [x] Baselines: logistic regression and 1D CNN
- [x] LoRA fine-tuning of Nucleotide Transformer (code, with a Colab notebook)
- [x] Compare with CADD, REVEL and AlphaMissense, by gene and review status (code)
- [x] Small Streamlit demo (code)
- [ ] Full training run and final results

More detail in [`reports/PLAN.md`](reports/PLAN.md).

## Limitations

- This is a learning project, not a clinical tool. Please don't use it for clinical decisions.
- ClinVar is biased toward well-studied genes and toward people of European ancestry.
- Only missense SNVs. No indels or structural variants.
- Sequence alone leaves out things clinical tools use, like protein structure and conservation scores.

## Data

All public: [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/), [Ensembl GRCh38](https://www.ensembl.org/), [dbNSFP](https://www.dbnsfp.org/) via [MyVariant.info](https://myvariant.info/) for CADD, REVEL and AlphaMissense scores, and [gnomAD](https://gnomad.broadinstitute.org/) for allele frequencies.

## License

MIT
