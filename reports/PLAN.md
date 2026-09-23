# DeepVariantEffect – Project Plan

## Phase 1 – Scaffold ✅
- GitHub repository, MIT license
- uv environment with core, ML, app, and dev extras
- Ruff linting and formatting, pre-commit hooks
- GitHub Actions CI running lint and tests
- Smoke test

## Phase 2 – Data
- Download the ClinVar GRCh38 VCF and the GRCh38 primary assembly FASTA
- Keep single-nucleotide variants annotated as missense
- Labels: Pathogenic / Likely pathogenic = 1, Benign / Likely benign = 0
- Drop conflicting interpretations and variants with no assertion criteria
- Extract ±512 bp reference context, build ref and alt sequences
- Split by chromosome to avoid leakage:
  - test: chr8, chr21
  - validation: chr10
  - train: all remaining autosomes and chrX
- Output: `data/processed/variants.parquet`
  - columns: chrom, pos, ref, alt, gene, review_status, label, split, ref_seq, alt_seq

## Phase 3 – Baselines
- Logistic regression on 6-mer count differences (alt − ref)
- 1D-CNN on one-hot encoded ref and alt sequences
- Report AUROC and AUPRC on the test chromosomes

## Phase 4 – Deep model
- Load InstaDeepAI/nucleotide-transformer-500m-human-ref
- Apply LoRA adapters to the attention layers (PEFT)
- Encode ref and alt, classify the embedding difference
- Train on a Google Colab T4 GPU, mixed precision, early stopping on validation AUPRC

## Phase 5 – Evaluation
- Metrics: AUROC, AUPRC, Brier score, reliability diagram
- Benchmark against CADD and REVEL from dbNSFP on the same variants
- Stratify by gene and by ClinVar review status
- Save all figures to `reports/figures/`

## Phase 6 – Demo
- Streamlit app: enter a variant → score, saliency map, nearest ClinVar neighbours
- Deploy on Hugging Face Spaces

## Phase 7 – Write-up
- `reports/REPORT.md` with methods, results, and figures
- Update README results table
- LinkedIn post announcing the project
