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


def test_wilson_ci_bounded_nondegenerate():
    lo0, hi0 = grade.wilson_ci(0, 15)
    assert lo0 == 0.0 and 0.0 < hi0 < 1.0
    lo1, hi1 = grade.wilson_ci(15, 15)
    assert hi1 == 1.0 and 0.0 < lo1 < 1.0
    lo, hi = grade.wilson_ci(5, 15)
    assert 0.0 < lo < 5 / 15 < hi < 1.0


def _manifest():
    return {
        "strata": {
            "web_search": {"frame_weight_mass": 2615.0},
            "fr_preamble": {"frame_weight_mass": 1719.0},
        }
    }


def _key_ws(n=4):
    return pl.DataFrame(
        {
            "label_row_id": [f"w{i}" for i in range(n)],
            "comment_id": [f"C{i}" for i in range(n)],
            "response_source": ["web_search"] * n,
            "response_sample_weight": [100.0, 1.0, 1.0, 1.0][:n],
        }
    )


def test_fn_rate_unweighted_and_weighted():
    key = _key_ws(4)
    # only the heavy row (w0) is a true "yes"
    labels = pl.DataFrame(
        {
            "label_row_id": ["w0", "w1", "w2", "w3"],
            "true_response_found": ["yes", "no", "no", "no"],
            "evidence_quote": ["q", "", "", ""],
            "true_agency_decision": ["accept", "", "", ""],
        }
    )
    stats = grade.compute_stats(labels, key, _manifest())
    ws = stats["strata"]["web_search"]
    assert ws["n"] == 4 and ws["yes"] == 1
    assert abs(ws["fn_unweighted"] - 0.25) < 1e-9
    # weighted: 100 / (100+1+1+1) = 0.9709...
    assert abs(ws["fn_weighted"] - 100 / 103) < 1e-9
    # projection = fn_weighted * frame_weight_mass
    assert abs(ws["projected_missed"] - (100 / 103) * 2615.0) < 1e-6


def test_weighted_equals_unweighted_under_uniform_weights():
    key = pl.DataFrame(
        {
            "label_row_id": ["w0", "w1", "w2", "w3"],
            "comment_id": ["C0", "C1", "C2", "C3"],
            "response_source": ["web_search"] * 4,
            "response_sample_weight": [5.0, 5.0, 5.0, 5.0],
        }
    )
    labels = pl.DataFrame(
        {
            "label_row_id": ["w0", "w1", "w2", "w3"],
            "true_response_found": ["yes", "yes", "no", "no"],
            "evidence_quote": ["q", "q", "", ""],
            "true_agency_decision": ["", "", "", ""],
        }
    )
    ws = grade.compute_stats(labels, key, _manifest())["strata"]["web_search"]
    assert abs(ws["fn_unweighted"] - ws["fn_weighted"]) < 1e-9


def test_contrast_present():
    key = _key_ws(2)
    labels = pl.DataFrame(
        {
            "label_row_id": ["w0", "w1"],
            "true_response_found": ["yes", "no"],
            "evidence_quote": ["q", ""],
            "true_agency_decision": ["", ""],
        }
    )
    stats = grade.compute_stats(labels, key, _manifest())
    assert "contrast_web_minus_fr" in stats


def _stats_fixture():
    return {
        "strata": {
            "web_search": {"n": 15, "yes": 4, "uncertain": 1, "fn_unweighted": 4 / 15,
                           "fn_unweighted_ci95": [0.1, 0.5], "fn_weighted": 0.30,
                           "frame_weight_mass": 2615.0, "projected_missed": 784.5},
            "fr_preamble": {"n": 15, "yes": 2, "uncertain": 0, "fn_unweighted": 2 / 15,
                            "fn_unweighted_ci95": [0.03, 0.4], "fn_weighted": 0.13,
                            "frame_weight_mass": 1719.0, "projected_missed": 223.5},
        },
        "contrast_web_minus_fr": 4 / 15 - 2 / 15,
    }


def test_report_embeds_honesty_caveats():
    md = grade.render_report(_stats_fixture(), n_per_stratum=15)
    assert "directional" in md.lower()  # not-publishable caveat
    assert "final rule provably exists" in md.lower()  # frame caveat
    assert "web_search" in md and "fr_preamble" in md


def test_write_results_creates_both_files(tmp_path):
    grade.write_results(_stats_fixture(), tmp_path, n_per_stratum=15)
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "results.md").exists()
