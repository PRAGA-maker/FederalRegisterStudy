from __future__ import annotations

import polars as pl
import pytest

from stratification_scripts import config
from stratification_scripts.goldset import sample


def test_goldset_dir_is_repo_root_goldset():
    d = config.get_goldset_dir()
    assert d == config.get_project_root() / "goldset"
    assert d.is_dir()  # mkdir side effect


def test_goldset_seed_path_joins_id():
    p = config.get_goldset_seed_path("2026-07-17-ce44ac5")
    assert p == config.get_goldset_dir() / "2026-07-17-ce44ac5"


def _agency_responses_fixture() -> pl.DataFrame:
    # Every combination that could be in/out of the frame.
    return pl.DataFrame(
        {
            "comment_id": [f"C{i}" for i in range(6)],
            "document_number": [f"D{i}" for i in range(6)],
            "lifecycle_stage": [
                "FINAL_EFFECTIVE",  # in (web_search)
                "FINAL_EFFECTIVE",  # in (fr_preamble)
                "FINAL_EFFECTIVE",  # out: response_found=yes
                "NPRM_CLOSED",      # out: wrong stage
                "FINAL_EFFECTIVE",  # out: response_found=uncertain
                "NO_RIN",           # out: wrong stage
            ],
            "response_found": ["no", "no", "yes", "no", "uncertain", "no"],
            "response_source": [
                "web_search", "fr_preamble", "web_search",
                "web_search", "fr_preamble", "web_search",
            ],
            "response_sample_weight": ["10.0", "5.0", "3.0", "1.0", "2.0", "4.0"],
        }
    )


def test_frame_filter_selects_only_checkable_no_rows():
    df = _agency_responses_fixture()
    frame = sample.frame_from_agency_responses(df)
    assert sorted(frame["comment_id"].to_list()) == ["C0", "C1"]
    assert set(frame["response_source"].to_list()) == {"web_search", "fr_preamble"}
