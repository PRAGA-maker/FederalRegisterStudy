from __future__ import annotations

from datetime import datetime, timezone

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


def _frame_fixture(per_source: int = 40) -> pl.DataFrame:
    rows = []
    for src in ("web_search", "fr_preamble"):
        for i in range(per_source):
            rows.append(
                {
                    "comment_id": f"{src}-{i:03d}",
                    "document_number": f"D-{src}-{i:03d}",
                    "response_source": src,
                    "response_sample_weight": str(1.0 + i),
                }
            )
    return pl.DataFrame(rows)


def test_draw_is_seeded_deterministic():
    frame = _frame_fixture()
    a = sample.draw_sample(frame, snapshot_id="S", seed=7, n=15, overlap=10)
    b = sample.draw_sample(frame, snapshot_id="S", seed=7, n=15, overlap=10)
    assert a["comment_id"].to_list() == b["comment_id"].to_list()


def test_draw_different_seed_differs():
    frame = _frame_fixture()
    a = sample.draw_sample(frame, snapshot_id="S", seed=7, n=15, overlap=10)
    c = sample.draw_sample(frame, snapshot_id="S", seed=8, n=15, overlap=10)
    assert set(a["comment_id"].to_list()) != set(c["comment_id"].to_list())


def test_draw_allocates_n_per_stratum():
    frame = _frame_fixture()
    s = sample.draw_sample(frame, snapshot_id="S", seed=7, n=15, overlap=10)
    counts = s.group_by("response_source").len().sort("response_source")
    assert counts["len"].to_list() == [15, 15]  # fr_preamble, web_search


def test_draw_raises_when_n_exceeds_stratum():
    frame = _frame_fixture(per_source=5)
    with pytest.raises(ValueError, match="only 5"):
        sample.draw_sample(frame, snapshot_id="S", seed=7, n=15, overlap=3)


def test_label_row_id_opaque_unique_stable():
    frame = _frame_fixture()
    s = sample.draw_sample(frame, snapshot_id="S", seed=7, n=15, overlap=10)
    ids = s["label_row_id"].to_list()
    assert len(ids) == len(set(ids))  # unique
    # opaque: encodes neither the source string nor a running index
    assert all("web_search" not in i and "fr_preamble" not in i for i in ids)
    # stable for (seed, snapshot)
    again = sample.draw_sample(frame, snapshot_id="S", seed=7, n=15, overlap=10)
    m1 = dict(zip(s["comment_id"].to_list(), s["label_row_id"].to_list()))
    m2 = dict(zip(again["comment_id"].to_list(), again["label_row_id"].to_list()))
    assert m1 == m2


def test_draw_interleaves_strata():
    frame = _frame_fixture()
    s = sample.draw_sample(frame, snapshot_id="S", seed=7, n=15, overlap=10)
    srcs = s["response_source"].to_list()
    runs = 1 + sum(1 for a, b in zip(srcs, srcs[1:]) if a != b)
    assert runs > 2  # not a single web-block then fr-block (which would be runs == 2)


def test_overlap_flag_count():
    frame = _frame_fixture()
    s = sample.draw_sample(frame, snapshot_id="S", seed=7, n=15, overlap=10)
    assert s["overlap_candidate"].sum() == 10


def test_sample_manifest_records_frame_mass_and_sampled_rows():
    frame = _frame_fixture()
    s = sample.draw_sample(frame, snapshot_id="S", seed=7, n=15, overlap=10)
    moment = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
    m = sample.build_sample_manifest(
        frame, s, snapshot_id="S", seed=7, n=15, overlap=10, moment=moment
    )
    assert m["snapshot_id"] == "S" and m["seed"] == 7
    assert m["strata"]["web_search"]["frame_rows"] == 40
    # frame weight mass = sum of (1.0 + i) for i in 0..39 = 40 + 780 = 820
    assert m["strata"]["web_search"]["frame_weight_mass"] == 820.0
    assert len(m["sampled"]) == 30
    assert m["sampled"][0]["label_row_id"] in s["label_row_id"].to_list()


def test_make_seed_id_uses_date_and_snapshot_short():
    moment = datetime(2026, 7, 17, 3, 0, tzinfo=timezone.utc)
    assert sample.make_seed_id("2026-07-15-ce44ac5", moment) == "2026-07-17-ce44ac5"
