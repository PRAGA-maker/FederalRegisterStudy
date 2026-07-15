# Frozen CSV Snapshots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A stamped, immutable, self-verifying local snapshot of the 2014+2024 pipeline CSVs, addressable by ID, that the gold-set/eval work labels against.

**Architecture:** A standalone module `stratification_scripts/freeze.py` (run as `python -m stratification_scripts.freeze`), deliberately NOT wired into the pipeline CLI. Pure primitives (hash, row-count, file-list, pointer-detect) compose into `create_snapshot` / `verify_snapshot` / `list_snapshots`, each dependency-injectable so tests run on tiny fixtures. Frozen CSV bytes stay local (gitignored); only `manifest.json` + tooling are committed.

**Tech Stack:** Python 3, polars (row counts), stdlib (`hashlib`, `shutil`, `subprocess`, `argparse`, `json`, `os`, `stat`), pytest.

## Global Constraints

- **Not a pipeline step.** `freeze.py` must NOT be imported by `stratification_scripts/cli.py` and must NOT be referenced by `run_all_years.sh` / `run_single_year.sh`. Structural separation is the "outside the re-runnable path" guarantee.
- **Never commit frozen CSV bytes.** `.gitignore` excludes `frozen/**/*.csv` (glob targets CSVs, not `frozen/`, so `manifest.json` stays committable). Committing them would bill the parent repository's LFS quota for zero gain.
- **Row counts via polars**, not `wc -l` — free-text fields have embedded newlines; the manifest count must equal downstream `polars` record count.
- **Push-safety:** no teammate names / status, no audit-doc or finding-ID references in any committed file (code, comments, commit messages).
- **Snapshot ID** = `<YYYY-MM-DD>-<git-short-sha>[-<label>]`.
- **Frozen file permissions:** `chmod 444` after copy (soft tamper guard).
- **Do not push `dev`** — all commits local until a separate go-ahead.
- Follow repo test convention: tests in `tests/` at repo root; `python -m pytest` from repo root (conftest adds the package to `sys.path`).

---

## File Structure

- **Create** `stratification_scripts/freeze.py` — the whole feature: types (`FrozenFile`, `GitInfo`, `VerifyReport`), primitives, the three operations, and the argparse `main`.
- **Modify** `stratification_scripts/config.py` — add `get_frozen_dir()` + `get_frozen_snapshot_path()`.
- **Modify** `.gitignore` — exclude `frozen/**/*.csv`.
- **Create** `tests/test_freeze.py` — all unit tests, run on fixture CSVs in `tmp_path`.

---

### Task 1: Config helpers + gitignore

**Files:**
- Modify: `stratification_scripts/config.py` (append two functions after `get_lifecycle_csv_path`, ~line 333)
- Modify: `.gitignore`
- Test: `tests/test_freeze.py` (new)

**Interfaces:**
- Consumes: `config.get_project_root()` (existing → repo root).
- Produces: `config.get_frozen_dir() -> Path` (repo-root `frozen/`, mkdir'd); `config.get_frozen_snapshot_path(snapshot_id: str) -> Path`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_freeze.py`:

```python
from pathlib import Path

from stratification_scripts import config


def test_get_frozen_dir_is_repo_root_frozen():
    frozen = config.get_frozen_dir()
    assert frozen == config.get_project_root() / "frozen"
    assert frozen.is_dir()  # mkdir side effect


def test_get_frozen_snapshot_path_joins_id():
    p = config.get_frozen_snapshot_path("2026-07-15-abc1234")
    assert p == config.get_frozen_dir() / "2026-07-15-abc1234"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_freeze.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'get_frozen_dir'`

- [ ] **Step 3: Implement the helpers**

Append to `stratification_scripts/config.py`:

```python
def get_frozen_dir() -> Path:
    """
    Get the repo-root frozen/ directory holding immutable CSV snapshots.

    Deliberately at the project root (one level above the package data dirs)
    for extra separation from the re-runnable pipeline path.

    Returns:
        Path to frozen/ directory.

    Side Effects:
        Creates the directory if it doesn't exist.
    """
    path = get_project_root() / "frozen"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_frozen_snapshot_path(snapshot_id: str) -> Path:
    """
    Get the directory for a specific frozen snapshot.

    Args:
        snapshot_id: Snapshot identifier, e.g. "2026-07-15-17958e6".

    Returns:
        Path to frozen/<snapshot_id>/.
    """
    return get_frozen_dir() / snapshot_id
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_freeze.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Add the gitignore rule**

Append to `.gitignore`:

```
# Frozen snapshot CSV bytes stay local — manifests are committed, bytes are not
frozen/**/*.csv
```

- [ ] **Step 6: Verify the gitignore rule targets CSVs but not manifests**

Run:
```bash
git check-ignore -v frozen/2026-07-15-x/makeup/data/agency_responses_2024.csv; echo "csv rc=$?"
git check-ignore -v frozen/2026-07-15-x/manifest.json; echo "manifest rc=$?"
```
Expected: the `.csv` line prints a match (`csv rc=0`); the `manifest.json` line prints nothing (`manifest rc=1`).

- [ ] **Step 7: Commit**

```bash
git add stratification_scripts/config.py .gitignore tests/test_freeze.py
git commit -m "feat(freeze): frozen-dir config helpers + gitignore frozen CSV bytes"
```

---

### Task 2: Pure primitives (pointer-detect, file record, file list)

**Files:**
- Create: `stratification_scripts/freeze.py`
- Test: `tests/test_freeze.py`

**Interfaces:**
- Consumes: `config.get_package_root()` and the six path helpers (`get_agency_responses_path`, `get_makeup_results_path`, `get_comments_raw_path`, `get_lifecycle_csv_path`, `get_fr_csv_path`, `get_makeup_data_path`).
- Produces:
  - `FrozenFile` (frozen dataclass: `source: Path`, `rel: str`).
  - `is_lfs_pointer_stub(source: Path) -> bool`.
  - `compute_file_record(source: Path, rel: str) -> dict` → `{"path", "sha256", "row_count", "byte_size"}`.
  - `snapshot_file_list(years=DEFAULT_YEARS) -> list[FrozenFile]`; `DEFAULT_YEARS = (2014, 2024)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_freeze.py`:

```python
import polars as pl

from stratification_scripts import freeze


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_freeze.py -v -k "pointer or record or file_list or row_count"`
Expected: FAIL with `ModuleNotFoundError: No module named 'stratification_scripts.freeze'`

- [ ] **Step 3: Create the module with the primitives**

Create `stratification_scripts/freeze.py`:

```python
"""
Frozen CSV snapshots — stamped, immutable, self-verifying copies of the
2014+2024 pipeline CSVs, addressable by ID for the gold-set / eval work.

Run as:  python -m stratification_scripts.freeze <create|verify|list> ...

Deliberately standalone: NOT imported by cli.py, NOT a pipeline step.
Frozen CSV bytes stay local (gitignored); only manifest.json is committed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from stratification_scripts import config

DEFAULT_YEARS = (2014, 2024)

_LFS_POINTER_SIGNATURE = b"version https://git-lfs.github.com/spec/v1"

# Ordered so the snapshot subtree mirrors the live layout.
_PATH_HELPERS = (
    config.get_agency_responses_path,
    config.get_makeup_results_path,
    config.get_comments_raw_path,
    config.get_lifecycle_csv_path,
    config.get_fr_csv_path,
    config.get_makeup_data_path,
)


@dataclass(frozen=True)
class FrozenFile:
    """A source CSV and its path relative to the snapshot dir."""

    source: Path
    rel: str


def is_lfs_pointer_stub(source: Path) -> bool:
    """True if `source` is an unmaterialized Git-LFS pointer, not real data."""
    with open(source, "rb") as fh:
        head = fh.read(len(_LFS_POINTER_SIGNATURE))
    return head == _LFS_POINTER_SIGNATURE


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_csv_records(path: Path) -> int:
    """Parsed CSV record count (header excluded), matching downstream polars.

    infer_schema_length=0 reads every column as a string: fast, and immune to
    dtype-inference errors on messy real data. Row count is dtype-independent.
    """
    return pl.read_csv(path, infer_schema_length=0).height


def compute_file_record(source: Path, rel: str) -> dict:
    """sha256 + parsed row count + byte size for one file."""
    return {
        "path": rel,
        "sha256": _sha256(source),
        "row_count": _count_csv_records(source),
        "byte_size": source.stat().st_size,
    }


def snapshot_file_list(years=DEFAULT_YEARS) -> list[FrozenFile]:
    """The exact CSV set to freeze, derived from config path helpers.

    Relative paths mirror the live subtree (makeup/data/…, output/…) by taking
    each source path relative to the package root.
    """
    root = config.get_package_root()
    files: list[FrozenFile] = []
    for year in years:
        for helper in _PATH_HELPERS:
            src = helper(year)
            files.append(FrozenFile(source=src, rel=src.relative_to(root).as_posix()))
    return files
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_freeze.py -v -k "pointer or record or file_list or row_count"`
Expected: PASS (all selected pass)

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/freeze.py tests/test_freeze.py
git commit -m "feat(freeze): pure primitives — pointer detect, file record, file list"
```

---

### Task 3: `create_snapshot` (atomic staging + all guards)

**Files:**
- Modify: `stratification_scripts/freeze.py`
- Test: `tests/test_freeze.py`

**Interfaces:**
- Consumes: `FrozenFile`, `is_lfs_pointer_stub`, `compute_file_record`, `snapshot_file_list`, `config.get_frozen_dir`, `config.get_project_root`.
- Produces:
  - `GitInfo` (frozen dataclass: `commit: str`, `short: str`, `dirty: bool`).
  - `read_git_info(source_paths) -> GitInfo`.
  - `create_snapshot(years=DEFAULT_YEARS, label=None, *, frozen_dir=None, resolver=snapshot_file_list, git_info=None, moment=None) -> dict` (returns the manifest dict).
  - Module-level `_write_manifest(path, manifest)` (factored out so tests can inject a failure), `TOOL_VERSION = "1"`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_freeze.py`:

```python
from datetime import datetime, timezone

import pytest


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_freeze.py -v -k "create_snapshot"`
Expected: FAIL with `AttributeError: module ... has no attribute 'create_snapshot'`

- [ ] **Step 3: Implement create_snapshot + GitInfo + helpers**

Add to the imports block at the top of `stratification_scripts/freeze.py`:

```python
import json
import os
import shutil
import stat
import subprocess
from datetime import datetime, timezone
```

Append to `stratification_scripts/freeze.py`:

```python
TOOL_VERSION = "1"


@dataclass(frozen=True)
class GitInfo:
    commit: str
    short: str
    dirty: bool


def read_git_info(source_paths) -> GitInfo:
    """Read HEAD commit + whether the working tree is dirty for the given paths."""
    root = config.get_project_root()

    def _out(args) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    commit = _out(["rev-parse", "HEAD"])
    short = _out(["rev-parse", "--short", "HEAD"])
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *[str(p) for p in source_paths]],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    return GitInfo(commit=commit, short=short, dirty=bool(status))


def _iso_z(moment: datetime) -> str:
    return (
        moment.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def _force_remove(func, path, _exc):
    """rmtree onerror: clear read-only bit (chmod 444 copies) then retry."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def create_snapshot(
    years=DEFAULT_YEARS,
    label=None,
    *,
    frozen_dir=None,
    resolver=snapshot_file_list,
    git_info=None,
    moment=None,
) -> dict:
    """Create an immutable frozen snapshot; return its manifest dict.

    Validates all sources BEFORE staging (so a rejected freeze leaves nothing
    behind). Stages into frozen/.tmp-<id>/, writes manifest LAST, then atomically
    renames to frozen/<id>/.
    """
    frozen_dir = frozen_dir if frozen_dir is not None else config.get_frozen_dir()
    files = resolver(years)

    for f in files:
        if not f.source.exists():
            raise FileNotFoundError(f"source missing: {f.source} (run the pipeline first)")
        if is_lfs_pointer_stub(f.source):
            raise ValueError(f"source is an unmaterialized lfs pointer: {f.source} — run `git lfs pull` first")

    git_info = git_info if git_info is not None else read_git_info([f.source for f in files])
    moment = moment if moment is not None else datetime.now(timezone.utc)

    snapshot_id = f"{moment.date().isoformat()}-{git_info.short}"
    if label:
        snapshot_id += f"-{label}"

    target = frozen_dir / snapshot_id
    if target.exists():
        raise FileExistsError(f"snapshot already exists: {target}")

    staging = frozen_dir / f".tmp-{snapshot_id}"
    if staging.exists():
        shutil.rmtree(staging, onerror=_force_remove)
    staging.mkdir(parents=True)

    records = []
    for f in files:
        dest = staging / f.rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f.source, dest)
        records.append(compute_file_record(dest, f.rel))
        os.chmod(dest, 0o444)

    manifest = {
        "snapshot_id": snapshot_id,
        "created_at": _iso_z(moment),
        "source_git_commit": git_info.commit,
        "source_git_dirty": git_info.dirty,
        "years": list(years),
        "tool_version": TOOL_VERSION,
        "files": records,
    }
    _write_manifest(staging / "manifest.json", manifest)  # LAST — validity sentinel
    os.replace(staging, target)  # atomic on one filesystem
    return manifest
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_freeze.py -v -k "create_snapshot"`
Expected: PASS (all create_snapshot tests pass)

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/freeze.py tests/test_freeze.py
git commit -m "feat(freeze): create_snapshot with atomic staging, pointer/missing/overwrite guards"
```

---

### Task 4: `verify_snapshot` + `list_snapshots`

**Files:**
- Modify: `stratification_scripts/freeze.py`
- Test: `tests/test_freeze.py`

**Interfaces:**
- Consumes: `_sha256`, `_count_csv_records`, `create_snapshot` (tests), `config.get_frozen_dir`.
- Produces:
  - `VerifyReport` (dataclass: `ok: bool`, `snapshot_id: str`, `problems: list[str]`, `checked: int`).
  - `verify_snapshot(snapshot_id, *, frozen_dir=None) -> VerifyReport`.
  - `list_snapshots(*, frozen_dir=None) -> list[dict]` (each: `snapshot_id`, `created_at`, `source_git_commit`, `file_count`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_freeze.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_freeze.py -v -k "verify or list_snapshots"`
Expected: FAIL with `AttributeError: module ... has no attribute 'verify_snapshot'`

- [ ] **Step 3: Implement verify_snapshot + list_snapshots**

Append to `stratification_scripts/freeze.py`:

```python
@dataclass
class VerifyReport:
    ok: bool
    snapshot_id: str
    problems: list
    checked: int


def verify_snapshot(snapshot_id, *, frozen_dir=None) -> VerifyReport:
    """Re-hash + re-count every file in a snapshot against its manifest."""
    frozen_dir = frozen_dir if frozen_dir is not None else config.get_frozen_dir()
    snap = frozen_dir / snapshot_id
    manifest_path = snap / "manifest.json"

    if not manifest_path.exists():
        return VerifyReport(False, snapshot_id, [f"manifest.json missing at {snap}"], 0)
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return VerifyReport(False, snapshot_id, [f"manifest unreadable: {exc}"], 0)

    problems: list[str] = []
    for rec in manifest["files"]:
        dest = snap / rec["path"]
        if not dest.exists():
            problems.append(f"MISSING {rec['path']}")
            continue
        if dest.stat().st_size != rec["byte_size"]:
            problems.append(f"SIZE {rec['path']}")
        if _sha256(dest) != rec["sha256"]:
            problems.append(f"SHA256 {rec['path']}")
        if _count_csv_records(dest) != rec["row_count"]:
            problems.append(f"ROWCOUNT {rec['path']}")

    return VerifyReport(not problems, snapshot_id, problems, len(manifest["files"]))


def list_snapshots(*, frozen_dir=None) -> list[dict]:
    """Summaries of valid snapshots; skips .tmp-* and manifest-less dirs."""
    frozen_dir = frozen_dir if frozen_dir is not None else config.get_frozen_dir()
    out: list[dict] = []
    if not frozen_dir.exists():
        return out
    for child in sorted(frozen_dir.iterdir()):
        if not child.is_dir() or child.name.startswith(".tmp-"):
            continue
        manifest_path = child / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        out.append(
            {
                "snapshot_id": manifest.get("snapshot_id", child.name),
                "created_at": manifest.get("created_at"),
                "source_git_commit": manifest.get("source_git_commit"),
                "file_count": len(manifest.get("files", [])),
            }
        )
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_freeze.py -v -k "verify or list_snapshots"`
Expected: PASS (all verify/list tests pass)

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/freeze.py tests/test_freeze.py
git commit -m "feat(freeze): verify_snapshot + list_snapshots"
```

---

### Task 5: CLI shell + end-to-end acceptance

**Files:**
- Modify: `stratification_scripts/freeze.py` (add `main` + `__main__` guard)
- Test: `tests/test_freeze.py`

**Interfaces:**
- Consumes: `create_snapshot`, `verify_snapshot`, `list_snapshots`, `DEFAULT_YEARS`.
- Produces: `main(argv=None) -> int` (exit code: 0 ok, 1 failure); `python -m stratification_scripts.freeze` entrypoint.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_freeze.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_freeze.py -v -k "main"`
Expected: FAIL with `AttributeError: module ... has no attribute 'main'`

- [ ] **Step 3: Implement main + __main__ guard**

Add to the imports block of `stratification_scripts/freeze.py`:

```python
import argparse
import sys
```

Append to `stratification_scripts/freeze.py`:

```python
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="freeze",
        description="Create/verify/list immutable frozen snapshots of the pipeline CSVs.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="Create a new frozen snapshot.")
    p_create.add_argument(
        "--year", type=int, action="append", dest="years",
        help="Year to freeze (repeatable). Default: 2014 and 2024.",
    )
    p_create.add_argument("--label", default=None, help="Optional snapshot-id suffix.")

    p_verify = sub.add_parser("verify", help="Verify a snapshot against its manifest.")
    p_verify.add_argument("snapshot_id")

    sub.add_parser("list", help="List valid snapshots.")

    args = parser.parse_args(argv)

    if args.cmd == "create":
        years = tuple(args.years) if args.years else DEFAULT_YEARS
        try:
            manifest = create_snapshot(years, args.label)
        except (FileNotFoundError, ValueError, FileExistsError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"created snapshot {manifest['snapshot_id']} ({len(manifest['files'])} files)")
        if manifest["source_git_dirty"]:
            print(
                "WARNING: frozen from a DIRTY tree — not restorable from history; "
                "back up the local CSVs out-of-band.",
                file=sys.stderr,
            )
        return 0

    if args.cmd == "verify":
        report = verify_snapshot(args.snapshot_id)
        for problem in report.problems:
            print(f"FAIL {problem}")
        print(f"{'OK' if report.ok else 'FAILED'}: {report.snapshot_id} ({report.checked} files)")
        return 0 if report.ok else 1

    if args.cmd == "list":
        for snap in list_snapshots():
            print(
                f"{snap['snapshot_id']}\t{snap['created_at']}\t"
                f"{snap['source_git_commit']}\t{snap['file_count']} files"
            )
        return 0

    return 1  # unreachable: subparser is required


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_freeze.py -v`
Expected: PASS (entire file green)

- [ ] **Step 5: Confirm freeze is NOT wired into the pipeline**

Run:
```bash
grep -rn "freeze" stratification_scripts/cli.py stratification_scripts/pipeline.py run_all_years.sh run_single_year.sh || echo "OK: not referenced by the pipeline"
```
Expected: `OK: not referenced by the pipeline`

- [ ] **Step 6: Commit**

```bash
git add stratification_scripts/freeze.py tests/test_freeze.py
git commit -m "feat(freeze): argparse CLI (create/verify/list) + module entrypoint"
```

- [ ] **Step 7: End-to-end acceptance on the real 2014+2024 data**

Preconditions: `git lfs` on PATH and the real CSVs materialized (not pointers). Run from a clean tree so `source_git_dirty` is `false`.

```bash
git lfs version                      # confirm LFS present
python -m stratification_scripts.freeze create
```
Expected: `created snapshot 2026-07-15-<sha> (12 files)`, no dirty warning.

```bash
SNAP=$(python -m stratification_scripts.freeze list | tail -1 | cut -f1)
python -m stratification_scripts.freeze verify "$SNAP"
```
Expected: `OK: 2026-07-15-<sha> (12 files)` (exit 0).

- [ ] **Step 8: Confirm git sees the manifest but not the CSV bytes**

Run:
```bash
git status --porcelain frozen/
```
Expected: exactly one untracked path — `frozen/<snap>/manifest.json` — and NO `.csv` lines (they are gitignored).

- [ ] **Step 9: Commit the manifest only**

```bash
git add frozen/*/manifest.json
git commit -m "chore(freeze): first frozen snapshot of 2014+2024 CSVs (manifest)"
```
Expected: only `manifest.json` committed; `git show --stat HEAD` lists no `.csv`.

---

## Self-Review

**Spec coverage:**
- Scope (12 files, config-derived) → Task 2 `snapshot_file_list` + test 8. ✅
- Layout / snapshot-ID scheme → Task 3 (`snapshot_id` construction). ✅
- Manifest fields + polars row_count → Task 2 `compute_file_record`, Task 3 manifest dict. ✅
- gitignore `frozen/**/*.csv` (not manifest) → Task 1 steps 5–6. ✅
- CLI create/verify/list, standalone → Task 5 + step 5 non-wiring check. ✅
- Pointer-stub guard → Task 3 test + `is_lfs_pointer_stub`. ✅
- Atomic staging, manifest-last, `.tmp-*` ignored → Task 3 (create), Task 4 (list skip), Task 5. ✅
- Refuse-overwrite, missing-source, dirty flag → Task 3. ✅
- chmod 444 → Task 3 test + impl. ✅
- verify detects sha/size/rowcount/missing/manifest-missing → Task 4. ✅
- Downstream helpers → Task 1. ✅
- Restoration story is documentation-only (no code this plan) — acceptance step 7 exercises the clean-freeze precondition. ✅

**Placeholder scan:** none — every code/test step is complete.

**Type consistency:** `FrozenFile(source, rel)`, `GitInfo(commit, short, dirty)`, `VerifyReport(ok, snapshot_id, problems, checked)`, manifest keys (`snapshot_id`, `created_at`, `source_git_commit`, `source_git_dirty`, `years`, `tool_version`, `files[].{path,sha256,row_count,byte_size}`) used identically across tasks. `create_snapshot`/`verify_snapshot`/`list_snapshots`/`main` signatures match between definitions and call sites. ✅
