"""Unit tests for the Phase 2 data pipeline (no network, no large files)."""

import pandas as pd
import pytest

from dve.data.build_dataset import assign_split, build
from dve.data.clinvar import first_gene, is_missense, label_from_clnsig, review_stars
from dve.data.reference import Reference, make_alt

# --- tiny synthetic reference: chromosome "1" = 40 bases -----------------------
CHROM1 = "ACGTACGTACGGGGCCCCAATTTTACGTACGTACGTACGT"

VCF_HEADER = """##fileformat=VCFv4.1
##fileDate=2026-09-01
##INFO=<ID=CLNSIG,Number=.,Type=String,Description="sig">
##INFO=<ID=CLNREVSTAT,Number=.,Type=String,Description="rev">
##INFO=<ID=CLNVC,Number=1,Type=String,Description="vc">
##INFO=<ID=MC,Number=.,Type=String,Description="mc">
##INFO=<ID=GENEINFO,Number=1,Type=String,Description="gene">
##contig=<ID=1>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
"""

MISSENSE = "MC=SO:0001583|missense_variant"
SNV = "CLNVC=single_nucleotide_variant"
ONE_STAR = "CLNREVSTAT=criteria_provided,_single_submitter"
ZERO_STAR = "CLNREVSTAT=no_assertion_criteria_provided"
SYNONYMOUS = "MC=SO:0001819|synonymous_variant"
OK = f"{ONE_STAR};{SNV};{MISSENSE}"


def _row(pos, vid, ref, alt, info):
    return f"1\t{pos}\t{vid}\t{ref}\t{alt}\t.\t.\t{info}\n"


@pytest.fixture
def toy_files(tmp_path):
    fasta = tmp_path / "ref.fa"
    fasta.write_text(">1\n" + CHROM1 + "\n")

    body = "".join(
        [
            # pos 15 = 'C' -> pathogenic, kept
            _row(15, 1, "C", "T", f"CLNSIG=Pathogenic;{OK};GENEINFO=GENEA:1"),
            # pos 21 = 'T' -> benign, kept
            _row(21, 2, "T", "G", f"CLNSIG=Benign;{OK};GENEINFO=GENEB:2"),
            # uncertain significance -> dropped
            _row(22, 3, "T", "A", f"CLNSIG=Uncertain_significance;{OK}"),
            # synonymous -> dropped
            _row(23, 4, "T", "C", f"CLNSIG=Benign;{ONE_STAR};{SNV};{SYNONYMOUS}"),
            # zero stars -> dropped
            _row(24, 5, "T", "A", f"CLNSIG=Benign;{ZERO_STAR};{SNV};{MISSENSE}"),
            # REF does not match the genome (pos 11 is 'G') -> dropped
            _row(11, 6, "A", "C", f"CLNSIG=Pathogenic;{OK}"),
            # too close to the chromosome start for flank=5 -> dropped
            _row(2, 7, "C", "A", f"CLNSIG=Pathogenic;{OK}"),
        ]
    )
    vcf = tmp_path / "clinvar.vcf"
    vcf.write_text(VCF_HEADER + body)
    return vcf, fasta


def test_label_from_clnsig():
    assert label_from_clnsig("Pathogenic") == 1
    assert label_from_clnsig("Likely_pathogenic") == 1
    assert label_from_clnsig("Pathogenic/Likely_pathogenic|other") == 1
    assert label_from_clnsig("Benign/Likely_benign") == 0
    assert label_from_clnsig("Uncertain_significance") is None
    assert label_from_clnsig("Conflicting_classifications_of_pathogenicity") is None
    assert label_from_clnsig(None) is None


def test_review_stars():
    assert review_stars("practice_guideline") == 4
    assert review_stars("reviewed_by_expert_panel") == 3
    assert review_stars("criteria_provided,_multiple_submitters,_no_conflicts") == 2
    assert review_stars("criteria_provided,_single_submitter") == 1
    assert review_stars("no_assertion_criteria_provided") == 0
    assert review_stars(None) == 0


def test_helpers():
    assert first_gene("BRCA1:672|NBR2:10230") == "BRCA1"
    assert first_gene(None) == ""
    assert is_missense("SO:0001583|missense_variant,SO:0001627|intron_variant")
    assert not is_missense("SO:0001819|synonymous_variant")


def test_reference_window(tmp_path):
    fasta = tmp_path / "ref.fa"
    fasta.write_text(">1\n" + CHROM1 + "\n")
    ref = Reference(str(fasta))
    w = ref.window("1", 15, flank=3)
    assert w == CHROM1[11:18]
    assert len(w) == 7 and w[3] == CHROM1[14]
    assert ref.window("1", 2, flank=3) is None  # runs off the start
    assert ref.window("2", 10, flank=3) is None  # unknown chromosome


def test_make_alt():
    assert make_alt("AAACAAA", 3, "T") == "AAATAAA"


def test_assign_split():
    splits = {"test": ["8", "21"], "val": ["10"]}
    assert assign_split("8", splits) == "test"
    assert assign_split("10", splits) == "val"
    assert assign_split("1", splits) == "train"


def test_build_end_to_end(toy_files):
    vcf, fasta = toy_files
    cfg = {
        "clinvar_vcf": str(vcf),
        "reference_fasta": str(fasta),
        "flank": 5,
        "min_review_stars": 1,
        "chromosomes": ["1"],
        "splits": {"test": ["8"], "val": ["10"]},
    }
    df, summary = build(cfg)

    assert isinstance(df, pd.DataFrame)
    assert list(df["variation_id"]) == ["1", "2"]
    assert list(df["label"]) == [1, 0]
    assert set(df["split"]) == {"train"}
    assert all(len(s) == 11 for s in df["ref_seq"])
    assert all(r[5] != a[5] for r, a in zip(df["ref_seq"], df["alt_seq"], strict=True))
    assert summary["dropped"]["ref_mismatch"] == 1
    assert summary["dropped"]["off_chromosome"] == 1
    assert summary["clinvar_release"] == "2026-09-01"
