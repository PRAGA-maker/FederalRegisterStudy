from pathlib import Path

import polars as pl

from stratification_scripts import config, freeze


def test_get_frozen_dir_is_repo_root_frozen():
    frozen = config.get_frozen_dir()
    assert frozen == config.get_project_root() / "frozen"
    assert frozen.is_dir()  # mkdir side effect


def test_get_frozen_snapshot_path_joins_id():
    p = config.get_frozen_snapshot_path("2026-07-15-abc1234")
    assert p == config.get_frozen_dir() / "2026-07-15-abc1234"


def _write_csv(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_is_lfs_pointer_stub_true_for_pointer(tmp_path):
    p = _write_csv(
        tmp_path / "ptr.csv",
        "version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\nsize 123\n",
    )
    assert freeze.is_lfs_pointer_stub(p) is True


def test_is_lfs_pointer_stub_false_for_real_csv(tmp_path):
    p = _write_csv(tmp_path / "real.csv", "a,b\n1,2\n")
    assert freeze.is_lfs_pointer_stub(p) is False


def test_compute_file_record_fields(tmp_path):
    p = _write_csv(tmp_path / "d.csv", "a,b\n1,2\n3,4\n")
    rec = freeze.compute_file_record(p, "sub/d.csv")
    assert rec["path"] == "sub/d.csv"
    assert rec["row_count"] == 2  # 2 data rows, header excluded
    assert rec["byte_size"] == p.stat().st_size
    assert len(rec["sha256"]) == 64


def test_compute_file_record_row_count_is_records_not_lines(tmp_path):
    # One data record whose quoted field contains a newline.
    p = _write_csv(tmp_path / "nl.csv", 'id,text\n1,"line one\nline two"\n')
    physical_lines = p.read_text().count("\n")
    rec = freeze.compute_file_record(p, "nl.csv")
    assert rec["row_count"] == 1
    assert physical_lines > 2  # header + 2 physical lines for the one record
    assert rec["row_count"] == pl.read_csv(p, infer_schema_length=0).height


def test_snapshot_file_list_twelve_expected_rels():
    files = freeze.snapshot_file_list((2014, 2024))
    rels = {f.rel for f in files}
    expected = {
        f"makeup/data/{name}_{year}.csv"
        for year in (2014, 2024)
        for name in ("agency_responses", "makeup_results", "comments_raw", "rin_lifecycle")
    } | {
        f"output/federal_register_{year}_comments.csv" for year in (2014, 2024)
    } | {
        f"output/makeup_data_{year}.csv" for year in (2014, 2024)
    }
    assert rels == expected
    assert len(files) == 12
