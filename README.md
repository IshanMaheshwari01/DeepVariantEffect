# DeepVariantEffect

**Deep learning for human variant pathogenicity prediction from DNA sequence.**

DeepVariantEffect fine-tunes a nucleotide language model (Nucleotide Transformer,
InstaDeepAI) on ClinVar-labelled missense single-nucleotide variants (SNVs) to
predict **pathogenic vs benign** directly from the ±512 bp reference-genome
context around each variant. Results are benchmarked against established
variant-effect predictors **CADD** and **REVEL** using AUROC and AUPRC, with
per-gene and per-review-status stratified evaluation.

An interactive Streamlit app lets a user enter a variant (for example
`chr7:117559590 G>A`) and receive a predicted pathogenicity score, a saliency
map over the surrounding sequence, and the most similar ClinVar variants.

---

## Why this project

Clinical variant interpretation still relies heavily on curated databases and
hand-crafted scores. Most variants seen in a patient's genome have never been
classified, and are reported as *Variants of Uncertain Significance (VUS)*.

Sequence-based deep learning models, pre-trained on the human reference genome,
offer a complementary signal that can generalise to variants not yet seen in
ClinVar. This project reproduces that idea end-to-end with a small, clean,
reproducible codebase and a public demo.

It is designed as the natural next step after germline variant calling
(see [gatk4-germline-pipeline](https://github.com/IshanMaheshwari01/gatk4-germline-pipeline)):
once a VCF is produced, *which of these variants actually matter?*

---

## Highlights

- **Foundation model fine-tuning** – LoRA fine-tuning of
  `InstaDeepAI/nucleotide-transformer-500m-human-ref` with Hugging Face PEFT
- **Honest baselines** – logistic regression on k-mer counts and a 1D-CNN on
  one-hot encoded sequence
- **Benchmarked against the field** – head-to-head comparison with CADD and
  REVEL scores on the same held-out variants
- **Leakage-aware evaluation** – train / validation / test split by chromosome
- **Explainability** – saliency maps showing which bases drive each prediction
- **Reproducible** – `uv` lockfile, deterministic seeds, YAML configs,
  pre-commit hooks, GitHub Actions CI
- **Deployable demo** – Streamlit app hosted on Hugging Face Spaces

---

## Method overview

ClinVar VCF ──► filter missense SNVs ──► label (Pathogenic = 1, Benign = 0)
│
GRCh38 FASTA ──► extract ±512 bp context ◄────┘
│
┌─────────────────────┴─────────────────────┐
▼ ▼
reference sequence alternate sequence
│ │
└──────► Nucleotide Transformer ◄───────────┘
(LoRA fine-tuned)
│
embedding difference (alt − ref)
│
classification head
│
P(pathogenic) ∈


---

## Results

> Results will be filled in as each phase completes.

| Model                          | AUROC | AUPRC | Notes                      |
|--------------------------------|:-----:|:-----:|----------------------------|
| Logistic regression (6-mers)   |   –   |   –   | Baseline                   |
| 1D-CNN (one-hot)               |   –   |   –   | Baseline                   |
| Nucleotide Transformer + LoRA  |   –   |   –   | Main model                 |
| CADD (PHRED)                   |   –   |   –   | External benchmark         |
| REVEL                          |   –   |   –   | External benchmark         |

Evaluation set: held-out chromosomes (chr8, chr21), unseen during training.

---

## Tech stack

| Area               | Tools                                                        |
|--------------------|--------------------------------------------------------------|
| Language           | Python 3.11                                                  |
| Deep learning      | PyTorch, Hugging Face Transformers, PEFT (LoRA), Accelerate  |
| Classical ML       | scikit-learn                                                 |
| Genomics I/O       | pysam, cyvcf2, pyfaidx, Biopython, bcftools, samtools        |
| Data               | pandas, PyArrow (Parquet)                                    |
| Visualisation      | Matplotlib, Seaborn, Plotly                                  |
| App                | Streamlit, Hugging Face Spaces                               |
| Engineering        | uv, Ruff, pre-commit, pytest, GitHub Actions                 |
| Compute            | WSL2 (development), Google Colab GPU (training)              |

---

## Repository layout

DeepVariantEffect/
├── src/dve/
│ ├── data/ # ClinVar and reference-genome ingestion, feature extraction
│ ├── models/ # baselines and transformer wrappers
│ ├── training/ # training loops and LoRA configuration
│ ├── eval/ # metrics, calibration, CADD / REVEL comparison
│ └── app/ # Streamlit demo
├── notebooks/ # exploratory analysis and Colab training notebooks
├── scripts/ # reproducible pipeline entry points
├── configs/ # YAML configs per experiment
├── tests/ # pytest unit and smoke tests
├── reports/
│ ├── PLAN.md # project roadmap
│ └── figures/ # generated plots
├── data/ # raw / interim / processed (git-ignored)
├── pyproject.toml
└── README.md


---

## Quickstart

```bash
# 1. Install uv (Linux / macOS / WSL)
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# 2. Clone the repository
git clone [https://github.com/IshanMaheshwari01/DeepVariantEffect.git](https://github.com/IshanMaheshwari01/DeepVariantEffect.git)
cd DeepVariantEffect

# 3. Install all dependencies
uv sync --all-extras

# 4. Run the tests
uv run pytest -q
```

Pipeline commands for data download, training, and evaluation will be added
here as each phase is completed.

---

## Roadmap

- [x] **Phase 1 – Scaffold:** repository, environment, CI, pre-commit
- [ ] **Phase 2 – Data:** ClinVar + GRCh38 → labelled Parquet dataset
- [ ] **Phase 3 – Baselines:** k-mer logistic regression and 1D-CNN
- [ ] **Phase 4 – Deep model:** LoRA fine-tuning of Nucleotide Transformer
- [ ] **Phase 5 – Evaluation:** AUROC / AUPRC / calibration, CADD and REVEL comparison
- [ ] **Phase 6 – Demo:** Streamlit app on Hugging Face Spaces
- [ ] **Phase 7 – Write-up:** technical report and blog post

Detailed plan: [`reports/PLAN.md`](reports/PLAN.md)

---

## Data sources

| Dataset      | Purpose                                   | Source                                      |
|--------------|-------------------------------------------|---------------------------------------------|
| ClinVar      | Variant pathogenicity labels              | https://www.ncbi.nlm.nih.gov/clinvar/       |
| GRCh38       | Reference genome sequence context         | https://www.ensembl.org/                    |
| dbNSFP       | Pre-computed CADD and REVEL scores        | https://www.dbnsfp.org/                     |
| gnomAD v4    | Population allele frequencies (QC)        | https://gnomad.broadinstitute.org/          |

All datasets are public. Download scripts live in `scripts/` and cache files
under `data/raw/`, which is git-ignored.

---

## Reproducibility

- Deterministic seeds for Python, NumPy, and PyTorch
- Every experiment driven by a YAML file in `configs/`
- Dependencies locked in `uv.lock`
- Lint and tests run automatically on every push via GitHub Actions
- Model checkpoints and metrics saved under `outputs/` (git-ignored)

---

## Limitations and ethics

DeepVariantEffect is a **research and educational project**. It is not a
medical device, has not been clinically validated, and **must not be used to
inform clinical decisions**.

Known limitations:

- ClinVar labels are biased towards well-studied genes and populations of
  European ancestry
- Only missense SNVs are modelled; indels and structural variants are out of scope
- Sequence context alone ignores protein structure, conservation, and
  expression information that clinical tools combine

---

## License

Released under the MIT License. See [`LICENSE`](LICENSE).

---

## Author

**Ishan Maheshwari** – MSc Genomics Data Science, University of Galway

[LinkedIn](https://www.linkedin.com/in/ishanmaheshwari2001) ·
[GitHub](https://github.com/IshanMaheshwari01)
