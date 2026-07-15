import os
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

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


def _fixture_resolver(tmp_path, files_spec):
    """Build a resolver returning FrozenFile(source, rel) for given specs."""
    made = []
    for rel, text in files_spec:
        src = _write_csv(tmp_path / "src" / rel, text)
        made.append(freeze.FrozenFile(source=src, rel=rel))
    return lambda years: made


def _clean_create(tmp_path, files_spec, *, frozen_dir, dirty=False, label=None):
    resolver = _fixture_resolver(tmp_path, files_spec)
    gi = freeze.GitInfo(commit="c0ffee123456", short="c0ffee1", dirty=dirty)
    moment = datetime(2026, 7, 15, 18, 22, 4, tzinfo=timezone.utc)
    return freeze.create_snapshot(
        (2024,), label, frozen_dir=frozen_dir, resolver=resolver, git_info=gi, moment=moment
    )


def test_create_snapshot_writes_matching_manifest(tmp_path):
    frozen = tmp_path / "frozen"
    spec = [("makeup/data/agency_responses_2024.csv", "a,b\n1,2\n3,4\n")]
    manifest = _clean_create(tmp_path, spec, frozen_dir=frozen)
    assert manifest["snapshot_id"] == "2026-07-15-c0ffee1"
    assert manifest["created_at"] == "2026-07-15T18:22:04Z"
    assert manifest["source_git_commit"] == "c0ffee123456"
    assert manifest["source_git_dirty"] is False
    rec = manifest["files"][0]
    snap = frozen / manifest["snapshot_id"]
    copied = snap / rec["path"]
    assert copied.exists()
    assert rec["sha256"] == freeze._sha256(copied)
    assert rec["row_count"] == 2


def test_create_snapshot_chmods_readonly(tmp_path):
    frozen = tmp_path / "frozen"
    spec = [("output/makeup_data_2024.csv", "x\n1\n")]
    manifest = _clean_create(tmp_path, spec, frozen_dir=frozen)
    copied = frozen / manifest["snapshot_id"] / "output/makeup_data_2024.csv"
    assert (copied.stat().st_mode & 0o222) == 0  # no write bits


def test_create_snapshot_refuses_overwrite(tmp_path):
    frozen = tmp_path / "frozen"
    spec = [("d.csv", "x\n1\n")]
    _clean_create(tmp_path, spec, frozen_dir=frozen)
    with pytest.raises(FileExistsError):
        _clean_create(tmp_path, spec, frozen_dir=frozen)


def test_create_snapshot_errors_on_missing_source(tmp_path):
    frozen = tmp_path / "frozen"
    gi = freeze.GitInfo("c", "c0ffee1", False)
    moment = datetime(2026, 7, 15, tzinfo=timezone.utc)
    resolver = lambda years: [freeze.FrozenFile(tmp_path / "nope.csv", "nope.csv")]
    with pytest.raises(FileNotFoundError):
        freeze.create_snapshot((2024,), frozen_dir=frozen, resolver=resolver, git_info=gi, moment=moment)
    assert not (frozen / "2026-07-15-c0ffee1").exists()


def test_create_snapshot_refuses_pointer_stub(tmp_path):
    frozen = tmp_path / "frozen"
    spec = [("d.csv", "version https://git-lfs.github.com/spec/v1\noid sha256:x\nsize 1\n")]
    with pytest.raises(ValueError, match="lfs"):
        _clean_create(tmp_path, spec, frozen_dir=frozen)
    assert not any(frozen.glob("2026-*")) if frozen.exists() else True


def test_create_snapshot_records_dirty_flag(tmp_path):
    frozen = tmp_path / "frozen"
    spec = [("d.csv", "x\n1\n")]
    manifest = _clean_create(tmp_path, spec, frozen_dir=frozen, dirty=True)
    assert manifest["source_git_dirty"] is True


def test_create_snapshot_atomic_on_failure(tmp_path, monkeypatch):
    frozen = tmp_path / "frozen"
    spec = [("d.csv", "x\n1\n")]
    resolver = _fixture_resolver(tmp_path, spec)
    gi = freeze.GitInfo("c", "c0ffee1", False)
    moment = datetime(2026, 7, 15, tzinfo=timezone.utc)

    def boom(path, manifest):
        raise RuntimeError("disk full")

    monkeypatch.setattr(freeze, "_write_manifest", boom)
    with pytest.raises(RuntimeError):
        freeze.create_snapshot((2024,), frozen_dir=frozen, resolver=resolver, git_info=gi, moment=moment)
    assert not (frozen / "2026-07-15-c0ffee1").exists()
    assert (frozen / ".tmp-2026-07-15-c0ffee1").exists()

    monkeypatch.undo()
    manifest = freeze.create_snapshot((2024,), frozen_dir=frozen, resolver=resolver, git_info=gi, moment=moment)
    assert (frozen / manifest["snapshot_id"] / "d.csv").exists()
    assert not (frozen / ".tmp-2026-07-15-c0ffee1").exists()


def test_verify_passes_clean(tmp_path):
    frozen = tmp_path / "frozen"
    spec = [("d.csv", 'id,text\n1,"a\nb"\n2,c\n')]  # embedded newline, exercises row_count
    manifest = _clean_create(tmp_path, spec, frozen_dir=frozen)
    report = freeze.verify_snapshot(manifest["snapshot_id"], frozen_dir=frozen)
    assert report.ok is True
    assert report.problems == []
    assert report.checked == 1


def test_verify_fails_on_flipped_byte(tmp_path):
    frozen = tmp_path / "frozen"
    spec = [("d.csv", "x\nhello\n")]
    manifest = _clean_create(tmp_path, spec, frozen_dir=frozen)
    target = frozen / manifest["snapshot_id"] / "d.csv"
    os.chmod(target, 0o644)
    target.write_text("x\nHELLO\n")
    report = freeze.verify_snapshot(manifest["snapshot_id"], frozen_dir=frozen)
    assert report.ok is False
    assert any("SHA256" in p for p in report.problems)


def test_verify_fails_on_missing_file(tmp_path):
    frozen = tmp_path / "frozen"
    spec = [("d.csv", "x\n1\n")]
    manifest = _clean_create(tmp_path, spec, frozen_dir=frozen)
    target = frozen / manifest["snapshot_id"] / "d.csv"
    os.chmod(target, 0o644)
    target.unlink()
    report = freeze.verify_snapshot(manifest["snapshot_id"], frozen_dir=frozen)
    assert report.ok is False
    assert any("MISSING" in p for p in report.problems)


def test_verify_fails_on_appended_row(tmp_path):
    frozen = tmp_path / "frozen"
    spec = [("d.csv", "x\n1\n")]
    manifest = _clean_create(tmp_path, spec, frozen_dir=frozen)
    target = frozen / manifest["snapshot_id"] / "d.csv"
    os.chmod(target, 0o644)
    with open(target, "a") as fh:
        fh.write("2\n")  # keep same first byte so size/sha catch it too; row_count differs
    report = freeze.verify_snapshot(manifest["snapshot_id"], frozen_dir=frozen)
    assert report.ok is False
    assert any("ROWCOUNT" in p for p in report.problems)


def test_verify_fails_when_manifest_missing(tmp_path):
    frozen = tmp_path / "frozen"
    (frozen / "2026-01-01-deadbee").mkdir(parents=True)
    report = freeze.verify_snapshot("2026-01-01-deadbee", frozen_dir=frozen)
    assert report.ok is False
    assert any("manifest" in p.lower() for p in report.problems)


def test_list_snapshots_skips_tmp_and_manifestless(tmp_path):
    frozen = tmp_path / "frozen"
    spec = [("d.csv", "x\n1\n")]
    manifest = _clean_create(tmp_path, spec, frozen_dir=frozen)
    (frozen / ".tmp-2099-01-01-abc").mkdir(parents=True)          # staging leftover
    (frozen / "2098-01-01-nomani").mkdir(parents=True)            # no manifest
    listed = freeze.list_snapshots(frozen_dir=frozen)
    ids = {s["snapshot_id"] for s in listed}
    assert ids == {manifest["snapshot_id"]}
    entry = listed[0]
    assert entry["file_count"] == 1
    assert entry["source_git_commit"] == "c0ffee123456"


def test_list_snapshots_empty_when_no_dir(tmp_path):
    assert freeze.list_snapshots(frozen_dir=tmp_path / "nope") == []


def test_main_verify_missing_returns_1(capsys):
    rc = freeze.main(["verify", "definitely-not-a-real-snapshot-id"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAILED" in out


def test_main_list_returns_0(capsys):
    rc = freeze.main(["list"])
    assert rc == 0


def test_main_requires_subcommand():
    with pytest.raises(SystemExit):
        freeze.main([])
