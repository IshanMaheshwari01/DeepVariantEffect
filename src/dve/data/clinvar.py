"""Parse the ClinVar VCF into a table of labelled missense SNVs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from cyvcf2 import VCF

PATHOGENIC = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
BENIGN = {"Benign", "Likely_benign", "Benign/Likely_benign"}

REVIEW_STARS = {
    "practice_guideline": 4,
    "reviewed_by_expert_panel": 3,
    "criteria_provided,_multiple_submitters,_no_conflicts": 2,
    "criteria_provided,_single_submitter": 1,
    "criteria_provided,_conflicting_classifications": 1,
    "criteria_provided,_conflicting_interpretations": 1,
}


@dataclass(frozen=True)
class ClinVarRecord:
    variation_id: str
    chrom: str
    pos: int  # 1-based, as in the VCF
    ref: str
    alt: str
    gene: str
    clnsig: str
    review_status: str
    stars: int
    label: int  # 1 = pathogenic, 0 = benign


def label_from_clnsig(clnsig: str | None) -> int | None:
    """Map a ClinVar CLNSIG string to 1 (pathogenic), 0 (benign) or None (drop)."""
    if not clnsig:
        return None
    primary = clnsig.split("|")[0].strip()
    if primary in PATHOGENIC:
        return 1
    if primary in BENIGN:
        return 0
    return None


def review_stars(revstat: str | None) -> int:
    """Convert a CLNREVSTAT string into ClinVar gold stars (0-4)."""
    if not revstat:
        return 0
    return REVIEW_STARS.get(revstat.strip(), 0)


def first_gene(geneinfo: str | None) -> str:
    """GENEINFO looks like 'BRCA1:672|NBR2:10230' -> 'BRCA1'."""
    if not geneinfo:
        return ""
    return geneinfo.split("|")[0].split(":")[0]


def is_missense(mc: str | None) -> bool:
    return bool(mc) and "missense_variant" in mc


def iter_clinvar(
    vcf_path: str,
    chromosomes: Iterable[str],
    min_stars: int = 1,
) -> Iterator[ClinVarRecord]:
    """Yield labelled missense SNVs from a ClinVar VCF."""
    keep_chroms = set(chromosomes)
    for v in VCF(vcf_path):
        if v.CHROM not in keep_chroms:
            continue
        if len(v.REF) != 1 or len(v.ALT) != 1 or len(v.ALT[0]) != 1:
            continue
        if v.REF not in "ACGT" or v.ALT[0] not in "ACGT":
            continue
        if v.INFO.get("CLNVC") != "single_nucleotide_variant":
            continue
        if not is_missense(v.INFO.get("MC")):
            continue
        clnsig = v.INFO.get("CLNSIG")
        label = label_from_clnsig(clnsig)
        if label is None:
            continue
        revstat = v.INFO.get("CLNREVSTAT")
        stars = review_stars(revstat)
        if stars < min_stars:
            continue
        yield ClinVarRecord(
            variation_id=str(v.ID),
            chrom=v.CHROM,
            pos=int(v.POS),
            ref=v.REF,
            alt=v.ALT[0],
            gene=first_gene(v.INFO.get("GENEINFO")),
            clnsig=clnsig,
            review_status=revstat or "",
            stars=stars,
            label=label,
        )


def clinvar_file_date(vcf_path: str) -> str:
    """Return the ##fileDate header value (ClinVar release date), if present."""
    for line in VCF(vcf_path).raw_header.splitlines():
        if line.startswith("##fileDate="):
            return line.split("=", 1)[1]
    return "unknown"
