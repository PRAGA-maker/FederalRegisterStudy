from __future__ import annotations

import polars as pl
import pytest

from stratification_scripts.goldset import grade


def _key() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "label_row_id": ["a1", "b2"],
            "comment_id": ["C1", "C2"],
            "response_source": ["web_search", "fr_preamble"],
            "response_sample_weight": [10.0, 5.0],
        }
    )


def _labels(**over) -> pl.DataFrame:
    base = {
        "label_row_id": ["a1", "b2"],
        "true_response_found": ["yes", "no"],
        "evidence_quote": ["see 89 FR 1", ""],
        "true_agency_decision": ["accept", ""],
    }
    base.update(over)
    return pl.DataFrame(base)


def test_validate_accepts_clean_labels():
    grade.validate_labels(_labels(), _key())  # no raise


def test_validate_rejects_unknown_id():
    bad = _labels(label_row_id=["a1", "zz"])
    with pytest.raises(ValueError, match="unknown label_row_id"):
        grade.validate_labels(bad, _key())


def test_validate_rejects_missing_row():
    bad = _labels(
        label_row_id=["a1"],
        true_response_found=["yes"],
        evidence_quote=["q"],
        true_agency_decision=["accept"],
    )
    with pytest.raises(ValueError, match="missing"):
        grade.validate_labels(bad, _key())


def test_validate_rejects_bad_enum():
    bad = _labels(true_response_found=["maybe", "no"])
    with pytest.raises(ValueError, match="true_response_found"):
        grade.validate_labels(bad, _key())


def test_validate_rejects_yes_without_evidence():
    bad = _labels(evidence_quote=["", ""])  # a1 is yes but has no evidence
    with pytest.raises(ValueError, match="evidence"):
        grade.validate_labels(bad, _key())
