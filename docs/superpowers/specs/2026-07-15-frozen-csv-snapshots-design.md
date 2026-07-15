# Frozen CSV Snapshots — Design

**Date:** 2026-07-15
**Status:** Approved, pre-implementation
**Owner:** Jonathan (eng seat)

## Problem

The gold-set / eval work (measuring the web-search false-negative rate, and everything
graded against it) needs to label rows in the 2014 + 2024 pipeline outputs. Those outputs
live at fixed, year-templated paths that the pipeline **overwrites in place** on every run:

- `stratification_scripts/config.py` writes all outputs via path helpers
  (`get_agency_responses_path(year)`, `get_makeup_results_path(year)`, …) that resolve to
  the *same* filename every run — e.g. `makeup/data/agency_responses_2024.csv`.
- A re-run therefore mutates the exact file the gold-set labels reference. Because mining is
  currently unseeded, a re-run does not even reproduce the same rows — labels would silently
  de-align from the data they were made against.

The CSVs are already git-LFS-tracked in history, so "commit X held content hash Y" is already
true. But git history alone does not solve this: the working-tree file at the pipeline's path
gets overwritten by a re-run or a checkout, and the downstream tooling needs a **self-describing
stamp** it can cite by ID — not git archaeology — to assert "labeled against exactly this data."

## Goal

A stamped, immutable, self-verifying snapshot of the 2014 + 2024 CSVs, living at a path the
pipeline never writes to, that downstream tooling (gold-set seed, Arvind annotation package)
references by an explicit snapshot ID.

**Acceptance:** the gold set can label against a fixed artifact whose integrity is verifiable
after the fact.

## Non-goals

- No `LATEST` / floating pointer — reproducibility means each consumer pins an explicit ID.
- No active pipeline write-guard — the pipeline structurally never writes under `frozen/`.
- No *routine* external (Drive/S3) store — the artifact stays in-repo next to the tooling that reads it.
  Exception (see Restoration): a snapshot frozen from a **dirty** tree cannot be rebuilt from git history,
  so its local CSVs become the sole copy — that single case gets a one-time out-of-band backup.
- No snapshot GC/pruning.
- Not a pipeline step — freezing is a deliberate manual act, never wired into `run_all_years.sh`.

## Scope of data frozen

All 2014 + 2024 CSVs — the full reproducibility anchor, not just the response-attribution file.
The exact file list is **derived from `config.py`'s path helpers** for each year in `{2014, 2024}`,
so it can never drift from what the pipeline actually produces. Per year (6 files × 2 years = 12):

| Helper | Live path (per year) |
|---|---|
| `get_agency_responses_path` | `makeup/data/agency_responses_{year}.csv` |
| `get_makeup_results_path` | `makeup/data/makeup_results_{year}.csv` |
| `get_comments_raw_path` | `makeup/data/comments_raw_{year}.csv` |
| `get_lifecycle_csv_path` | `makeup/data/rin_lifecycle_{year}.csv` |
| `get_fr_csv_path` | `output/federal_register_{year}_comments.csv` |
| `get_makeup_data_path` | `output/makeup_data_{year}.csv` |

The gold-set-critical file is `agency_responses_{year}.csv` — it carries `response_source`
(`fr_preamble` vs `web_search`, the grounded/web-search stratifier), `response_found`,
`agency_decision`, `response_text`, and `reasoning`.

## Layout & addressing

```
frozen/
  <snapshot-id>/
    manifest.json
    makeup/data/agency_responses_2014.csv
    makeup/data/agency_responses_2024.csv
    makeup/data/makeup_results_2014.csv
    makeup/data/makeup_results_2024.csv
    makeup/data/comments_raw_2014.csv
    makeup/data/comments_raw_2024.csv
    makeup/data/rin_lifecycle_2014.csv
    makeup/data/rin_lifecycle_2024.csv
    output/federal_register_2014_comments.csv
    output/federal_register_2024_comments.csv
    output/makeup_data_2014.csv
    output/makeup_data_2024.csv
```

- **Snapshot ID** = `<YYYY-MM-DD>-<git-short-sha>` (e.g. `2026-07-15-17958e6`). Date sorts
  chronologically; the short sha makes provenance visible in the path and avoids same-day
  collisions. An optional `--label` suffix is appended if provided: `<date>-<sha>-<label>`.
- The frozen subtree **mirrors the live subtree** (`makeup/data/`, `output/`) so relative
  paths are predictable from a consumer's point of view.
- **Frozen CSV bytes are NOT committed.** `.gitignore` excludes `frozen/**/*.csv` (the glob targets the
  CSVs, *not* `frozen/` itself, so `manifest.json` under each snapshot stays committable). Committing frozen
  CSVs via LFS is actively harmful here: LFS objects pushed from a fork count against the **parent
  repository's** quota; a tripped quota blocks LFS pushes for everyone, and the duplicate bytes
  buy nothing the manifest + git history don't already provide. **In git:** `manifest.json` + freeze tooling.
  **Local-only:** the frozen CSV bytes.

## manifest.json — the stamp

```json
{
  "snapshot_id": "2026-07-15-17958e6",
  "created_at": "2026-07-15T18:22:04Z",
  "source_git_commit": "17958e6...",
  "source_git_dirty": false,
  "years": [2014, 2024],
  "tool_version": "1",
  "files": [
    {
      "path": "makeup/data/agency_responses_2024.csv",
      "sha256": "220c236c...",
      "row_count": 12873,
      "byte_size": 1539593
    }
  ]
}
```

- `source_git_commit` / `source_git_dirty` capture the repo state the snapshot was cut from.
- `row_count` = **parsed CSV records** (via polars), NOT physical newline count, and excludes the header.
  `agency_responses_*.csv` carries free-text `response_text` / `reasoning` fields whose embedded newlines
  make `wc -l` diverge from the record count; the record count is what downstream `polars` `len()` agrees
  with, so that is what the manifest stores.
- `files[].path` is relative to the snapshot dir.

## CLI — `python -m stratification_scripts.freeze`

A standalone module `stratification_scripts/freeze.py` with its own argparse. **Not** imported
by `cli.py` and **not** referenced by `run_all_years.sh` / `run_single_year.sh`. That structural
separation is the "outside the re-runnable path" guarantee.

Subcommands:

- **`create [--year 2014 --year 2024] [--label X]`**
  - Resolves the file list from `config.py` helpers for the given years (default `{2014, 2024}`).
  - Errors if any source file is missing (fail loud — never freeze a partial set silently).
  - **Refuses any source that is an unmaterialized LFS pointer stub** — a file whose first bytes are
    `version https://git-lfs.github.com/spec/v1`. Such a file *exists* (so it passes the missing-file check)
    but is a ~130-byte pointer, not data; this repo produced exactly this failure mode on clone. The error
    tells the user to `git lfs pull` first.
  - Computes the snapshot ID from today's date + current git short sha (+ label).
  - **Refuses to overwrite an existing snapshot dir** (immutability).
  - **Atomic staging.** Copies into `frozen/.tmp-<id>/`, computes sha256 + row_count + byte_size per file,
    `chmod 444` the copies, writes `manifest.json` **last** (the validity sentinel), then `os.rename` the
    staging dir to `frozen/<id>/` (atomic on one filesystem). A mid-copy failure (disk full, interrupt) leaves
    only a `.tmp-<id>/` — which `list` / `verify` ignore and a re-run overwrites — never a half-populated
    `frozen/<id>/` that refuse-overwrite would then wedge.
  - If the working tree is dirty for any source file, records `source_git_dirty: true` and prints a warning
    (does not hard-fail — locally materialized data may legitimately differ from HEAD; the user decides).
    See Restoration for why a **clean** freeze is strongly preferred.

- **`verify <snapshot-id>`**
  - Re-reads `manifest.json`, re-hashes and re-counts every listed file.
  - Fails (nonzero exit) on: missing file, missing/unreadable `manifest.json`, sha256 mismatch,
    row_count mismatch, byte_size mismatch.
  - Prints a per-file OK/FAIL report and a summary line. Ignores `frozen/.tmp-*` staging dirs.

- **`list`**
  - Lists snapshot IDs under `frozen/` with their `created_at`, `source_git_commit`, file count.
  - Skips `frozen/.tmp-*` staging dirs and any dir lacking a readable `manifest.json`.

## Downstream interface

Two helpers added to `config.py`:

- `get_frozen_dir() -> Path` → `get_project_root() / "frozen"` (repo-root `frozen/`), created if absent.
  Note: the live data dirs sit *under* the package (`get_package_root()` = `stratification_scripts/`);
  `frozen/` deliberately sits one level up at the repo root, extra separation from the re-runnable path.
- `get_frozen_snapshot_path(snapshot_id: str) -> Path` → `get_frozen_dir() / snapshot_id`.

Gold-set tooling and the Arvind export read `get_frozen_snapshot_path(id) / "makeup/data/agency_responses_2024.csv"`
and **pin the explicit `id`** in their own config/output, so any produced labels name the snapshot
they were made against.

## Restoration (rebuilding a snapshot's CSVs from history)

Because the CSV bytes are gitignored, a fresh clone has only each snapshot's `manifest.json`. To rebuild the
actual files for a clean snapshot (`source_git_dirty == false`):

1. `git checkout <manifest.source_git_commit>` (or spin a worktree at it).
2. `git lfs pull` to materialize the live pipeline CSVs at that commit.
3. Copy them back under `frozen/<id>/` (a future `freeze restore <id>` helper can automate steps 1–3; for
   now it is manual).
4. `freeze verify <id>` to confirm the rebuilt bytes match the manifest hashes exactly.

**This path only holds when `source_git_dirty` is `false`.** A snapshot frozen from a dirty tree references
bytes that existed at no commit — its local `frozen/<id>/*.csv` are then the *sole* artifact. **Therefore
prefer freezing from a clean tree.** For any deliberately dirty freeze, back the local CSVs up out-of-band
(one-time Drive zip) — the single case where the "no routine external store" non-goal is relaxed.

## Module structure

`freeze.py` decomposes into pure, testable functions plus a thin CLI shell:

- `snapshot_file_list(years) -> list[FrozenFile]` — maps years → (source Path, relative dest path)
  via `config.py` helpers. Pure given config.
- `compute_file_record(source: Path, rel: str) -> dict` — sha256 (streamed in chunks) + row_count +
  byte_size. Pure. row_count via `polars.scan_csv(source).select(pl.len()).collect()` so it matches
  downstream polars reads, not `wc -l`.
- `is_lfs_pointer_stub(source: Path) -> bool` — true if the file's first bytes are the LFS pointer
  signature `version https://git-lfs.github.com/spec/v1`. Guards `create`.
- `create_snapshot(years, label, *, frozen_dir, source_resolver, git_info) -> manifest` — orchestrates
  copy + manifest write + chmod. Takes injected `frozen_dir` / resolver so tests use fixtures.
- `verify_snapshot(snapshot_id, *, frozen_dir) -> VerifyReport` — pure comparison, returns structured
  result; CLI maps to exit code.
- `list_snapshots(*, frozen_dir) -> list[...]`.
- `main(argv)` — argparse shell.

The dependency injection (`frozen_dir`, `source_resolver`, `git_info`) is what lets tests run against
small fixture CSVs in a temp dir instead of the 139 MB reals.

## Testing (TDD)

Fixtures: a temp dir with tiny synthetic CSVs standing in for the real files; injected `frozen_dir`.

1. `create_snapshot` writes a manifest whose sha256 / row_count / byte_size match independently
   recomputed values for each fixture file.
2. `verify_snapshot` passes clean immediately after `create_snapshot`.
3. `verify_snapshot` fails when one byte of a frozen file is flipped (sha256 + byte_size mismatch).
4. `verify_snapshot` fails when a frozen file is deleted (missing file).
5. `verify_snapshot` fails when a row is appended (row_count mismatch).
6. `create_snapshot` refuses to overwrite an existing snapshot dir (raises, leaves original intact).
7. `create_snapshot` errors when a source file is missing (no partial snapshot left behind).
8. `snapshot_file_list` returns exactly the 12 expected relative paths for `{2014, 2024}`.
9. Manifest carries all required top-level + per-file fields.
10. `chmod 444` applied to copied files (permission bits checked where the platform supports it).
11. `create` refuses a source that is an LFS pointer stub (fixture file whose first line is the pointer
    signature), with a clear "run git lfs pull" error — and leaves no snapshot behind.
12. `row_count` counts CSV records, not physical lines: a fixture with an embedded newline inside a quoted
    field has `manifest.row_count == polars height < physical line count`; `verify` still passes on it.
13. Atomic staging: an interrupted `create` (simulated by injecting a failure before the manifest write)
    leaves a `frozen/.tmp-<id>/` and **no** `frozen/<id>/`; a subsequent clean `create` succeeds.
14. `list` / `verify` ignore `.tmp-*` dirs and dirs lacking a readable `manifest.json`.

## Risks / notes

- **Zero added LFS/repo weight:** frozen CSV bytes are gitignored (see Layout), so a freeze costs no repo
  or LFS quota — critical because fork LFS pushes bill the parent repository's quota. Local disk only
  (~390 MB per snapshot).
- **Dirty-tree freeze:** allowed with a warning + `source_git_dirty` flag rather than a hard block, because
  locally materialized data may legitimately differ from HEAD; the flag preserves honesty. But a dirty
  snapshot is un-restorable from history (see Restoration) — hence the strong preference for clean freezes
  and the out-of-band backup carve-out.
