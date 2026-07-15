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
