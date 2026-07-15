"""
Frozen CSV snapshots — stamped, immutable, self-verifying copies of the
2014+2024 pipeline CSVs, addressable by ID for the gold-set / eval work.

Run as:  python -m stratification_scripts.freeze <create|verify|list> ...

Deliberately standalone: NOT imported by cli.py, NOT a pipeline step.
Frozen CSV bytes stay local (gitignored); only manifest.json is committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
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

    if "files" not in manifest:
        return VerifyReport(False, snapshot_id, ["manifest missing 'files' key"], 0)

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
        except (
            FileNotFoundError,
            ValueError,
            FileExistsError,
            subprocess.CalledProcessError,
        ) as exc:
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
