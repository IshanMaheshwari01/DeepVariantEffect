#!/usr/bin/env bash
# Everything except the transformer. Train that first or afterwards with
# `uv run python -m dve.training.train_nt`, then re-run the last four commands.
set -euo pipefail
[ -f data/processed/variants.parquet ] || {
  bash scripts/download_data.sh
  uv run python -m dve.data.build_dataset
}
[ -f reports/results/logreg_kmer.json ] || uv run python -m dve.training.train_logreg
[ -f reports/results/cnn_onehot.json ] || uv run python -m dve.training.train_cnn
uv run python -m dve.eval.external
uv run python scripts/compare_models.py
uv run python -m dve.eval.benchmark
uv run python scripts/make_report.py
uv run python scripts/export_app_data.py
