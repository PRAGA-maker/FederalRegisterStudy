"""
Frozen CSV snapshots — stamped, immutable, self-verifying copies of the
2014+2024 pipeline CSVs, addressable by ID for the gold-set / eval work.

Run as:  python -m stratification_scripts.freeze <create|verify|list> ...

Deliberately standalone: NOT imported by cli.py, NOT a pipeline step.
Frozen CSV bytes stay local (gitignored); only manifest.json is committed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
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
