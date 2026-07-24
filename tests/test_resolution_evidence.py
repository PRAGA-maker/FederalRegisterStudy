from stratification_scripts.makeup.fr_response_extractor import ResponseExtract
from stratification_scripts.resolution.evidence import (
    density_per_1k, response_evidence_from_extract,
)
from stratification_scripts.resolution.models import ResponseEvidence

DENSE = (
    "Comment: Several commenters argued the rule is too costly. "
    "Response: We disagree with the commenters and are adopting the provision. "
    "In response to comment, the agency considered the alternative. "
) * 20

SPARSE = ("This rule adopts technical amendments to the table of contents. " * 60)


def test_header_match_is_strong():
    ext = ResponseExtract("short text", "response_hd", "Comments and Responses", True, 5000, 10)
    assert response_evidence_from_extract(ext) is ResponseEvidence.STRONG


def test_dot_case_no_header_but_dense_text_is_strong():
    ext = ResponseExtract(DENSE, "suplinf_full", None, False, len(DENSE), len(DENSE))
    assert response_evidence_from_extract(ext) is ResponseEvidence.STRONG


def test_low_density_text_is_weak():
    ext = ResponseExtract(SPARSE, "suplinf_full", None, False, len(SPARSE), len(SPARSE))
    assert response_evidence_from_extract(ext) is ResponseEvidence.WEAK


def test_empty_extract_is_none():
    ext = ResponseExtract("", "no_preamble", None, False, 0, 0)
    assert response_evidence_from_extract(ext) is ResponseEvidence.NONE


def test_missing_extract_is_none():
    assert response_evidence_from_extract(None) is ResponseEvidence.NONE


def test_density_per_1k_is_a_rate_not_a_count():
    assert density_per_1k(DENSE) > density_per_1k(SPARSE)
    assert density_per_1k("") == 0.0
