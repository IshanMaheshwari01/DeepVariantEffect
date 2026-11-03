#!/usr/bin/env bash
# Download the public datasets used by DeepVariantEffect into data/raw/.
# Safe to re-run: existing files are skipped.
set -euo pipefail

RAW_DIR="data/raw"
mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

CLINVAR_URL="https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38"
ENSEMBL_URL="https://ftp.ebi.ac.uk/ensemblorg/pub/current_fasta/homo_sapiens/dna"
FASTA="Homo_sapiens.GRCh38.dna.primary_assembly.fa"

echo "==> ClinVar GRCh38 VCF (~190 MB)"
[ -f clinvar.vcf.gz ]     || wget -c -q --show-progress "$CLINVAR_URL/clinvar.vcf.gz"
[ -f clinvar.vcf.gz.tbi ] || wget -c -q --show-progress "$CLINVAR_URL/clinvar.vcf.gz.tbi"

echo "==> GRCh38 primary assembly FASTA (~900 MB download, ~3.1 GB unpacked)"
if [ ! -f "$FASTA" ]; then
  [ -f "$FASTA.gz" ] || wget -c -q --show-progress "$ENSEMBL_URL/$FASTA.gz"
  echo "    unpacking..."
  gzip -dc "$FASTA.gz" > "$FASTA"
  rm -f "$FASTA.gz"
fi

echo "==> Indexing FASTA"
# pyfaidx builds the index itself on first use if samtools is not installed
if [ ! -f "$FASTA.fai" ] && command -v samtools >/dev/null; then samtools faidx "$FASTA"; fi

echo
echo "Done. Files in $RAW_DIR:"
ls -lh
