# Gold-Set Seed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `goldset` tool that draws a reproducible stratified sample from the frozen 2024 data, exports a *blind* labeling packet (spreadsheet), and grades returned human labels into per-source web-search false-negative rates with honest uncertainty.

**Architecture:** A new `stratification_scripts/goldset/` subpackage mirroring `makeup/`, standalone like `freeze` (never wired into `cli.py`/`run_*.sh`). Three pure-logic modules — `sample` (frame filter, seeded stratified draw, manifest), `packet` (blind packet + hidden prediction key + link building), `grade` (validation, FN rates, Wilson CIs, HT-weighted projection, report) — behind an argparse `cli`. Sample/key/label CSVs are tiny, human-edited, irreplaceable, so they commit as **plain git blobs** via a `.gitattributes` override (opposite call to the frozen snapshot, same cost-vs-irreplaceability reasoning).

**Tech Stack:** Python 3.12, polars 1.36.1, stdlib `random`/`hashlib`/`argparse`/`json`, pytest. Design spec: `docs/superpowers/specs/2026-07-16-goldset-seed-design.md`.

## Global Constraints

- **Python invocation:** `uv` is NOT on PATH in this environment. Use `.venv/bin/python` for everything — tests (`.venv/bin/python -m pytest`) and CLI (`.venv/bin/python -m stratification_scripts.goldset ...`).
- **ruff line-length 120**, `target-version py310`, `from __future__ import annotations` at top of every new module (repo convention).
- **Frame is fixed and verified** against snapshot `2026-07-15-ce44ac5`: `lifecycle_stage == "FINAL_EFFECTIVE" AND response_found == "no" AND response_source IN ("web_search","fr_preamble")` → **378 rows** (web_search 150, fr_preamble 228; frame weight mass 4334.5 = 2615 web_search + 1719 fr_preamble). Enum literals are exact-case: `FINAL_EFFECTIVE`, `no`, `web_search`, `fr_preamble`.
- **Never leak the answer path.** The packet must contain NONE of: `response_found`, `agency_decision`, `reasoning`, `response_text`, `response_location`, `response_citation`, `rtc_document_id`, `tier2_acceptance_status`, `tier2_confidence`, `tier2_text_change_summary`, **`response_source`**. These live only in `prediction_key.csv`.
- **Never construct a URL from `docket_id`** — its values are prose (`Docket ID SBA-2024-0007`, `FAR Case 2019-015`). Ships as a text hint only.
- **Join guards (many-to-many fan-out hazard):** every join must be one-to-one. `comments_raw.comment_id` and `federal_register.document_number` are both unique in the 2024 snapshot (verified: 40482/40482 and 24433/24433), but the code dedupes `keep="first"` before joining AND asserts joined height == input height, so a future dirty CSV can never silently fan out.
- **Seeded, reproducible sampling only** — same `(seed, snapshot_id)` ⇒ identical sample forever. No unseeded RNG (that unreproducibility is exactly what the frozen snapshot exists to escape).
- **Snapshot read is pinned** — always `config.get_frozen_snapshot_path(<explicit id>)`, never a "latest" pointer.

### Real column names (verified against snapshot `2026-07-15-ce44ac5`)

- `agency_responses_2024.csv` (frame + hidden verdicts): `comment_id, document_number, agency, response_found, agency_decision, response_text, response_location, reasoning, processed_at, model, comment_text_length, has_attachment, lifecycle_stage, rin, response_sample_weight, response_source, response_citation, rtc_document_id, tier2_acceptance_status, tier2_confidence, tier2_text_change_summary`
- `comments_raw_2024.csv` (context): join on `comment_id` → `comment_text, organization, submitter_type`
- `federal_register_2024_comments.csv` (FR): join on `document_number` → `title, docket_id, final_action_citation, final_rule_document_number` (coverage on the frame: title 378/378, docket_id 378/378, final_action_citation 195/378, final_rule_document_number 162/378)

---

## Decision Ledger

> Appended as non-obvious implementation choices land (eng-seat standing duty; this is a repo session so the ledger lives here, not the vault). Each entry: what/why/rejected/what-would-reopen. Design-level decisions already live in the spec; this captures implementation-time calls.

- **(seed) `label_row_id` = first 12 hex of `sha256(f"{seed}|{snapshot_id}|{comment_id}")`.** Opaque (encodes neither stratum nor position), unique, and stable for a given `(seed, snapshot)` — satisfies the anti-anchoring requirement with a pure function needing no counter state. Rejected: sequential ids (encode position), stratum-prefixed ids (leak source). Reopen if: a 12-hex collision ever appears (birthday bound ~16M rows; we sample ~30) — the sampler asserts uniqueness and would fail loud.
- **(gitattributes) override is `goldset/**/*.csv -filter diff merge text`, NOT `-filter -diff -merge`.** First draft used `-diff -merge`, which unsets the diff/merge drivers and makes git treat the files as *binary* — reproducing the exact non-diffability the spec cites as an LFS harm. Corrected to `diff merge` (force textual diff + normal 3-way merge) + `text` (EOL normalize). Verified with `git check-attr` (filter unset; diff/merge/text set) AND by editing a cell and seeing a line-level `git diff`. The blanket `frozen/...` path still reports `filter: lfs`. Reopen if: a future goldset artifact is genuinely binary (none foreseen — they are all tiny CSV/JSON).
- **(interleave) canonicalize then seeded-shuffle, not shuffle-in-place.** `draw_sample` sorts the combined draw by `comment_id` before `rng.shuffle`, so the output order depends only on `(seed, snapshot)` and never on polars' concat/read order. Rejected: relying on polars `.sample`'s internal RNG (version-coupled, opaque). Reopen if: polars changes `DataFrame.__getitem__(list)` semantics (pinned at 1.36.1).
- **(grade) `uncertain` labels stay in the FN denominator.** FN = yes/n over all sampled rows in the stratum; an `uncertain` is neither a confirmed miss nor a confirmed correct-"no", so counting it in the denominator makes the headline FN a *lower* bound (conservative). The count is reported separately so the reader sees how many were unresolved. Rejected: dropping uncertains (inflates FN, over-claims the bias). Reopen if: uncertains become a large share — then a two-sided [drop | keep] band is worth reporting.

---

## File Structure

- `stratification_scripts/config.py` — **modify**: add `get_goldset_dir()` + `get_goldset_seed_path(seed_id)` (mirror the frozen helpers).
- `stratification_scripts/goldset/__init__.py` — **create**: package marker.
- `stratification_scripts/goldset/sample.py` — **create**: frame filter, `label_row_id`, seeded stratified draw, interleave, sample manifest.
- `stratification_scripts/goldset/packet.py` — **create**: link building, blind packet + hidden prediction key construction.
- `stratification_scripts/goldset/grade.py` — **create**: label loading/validation, Wilson CI, per-stratum FN stats, HT-weighted projection, report render, results writers.
- `stratification_scripts/goldset/cli.py` — **create**: argparse `sample | grade`, orchestration, overwrite guard.
- `stratification_scripts/goldset/__main__.py` — **create**: `python -m stratification_scripts.goldset` entrypoint.
- `.gitattributes` — **modify**: override the blanket `*.csv filter=lfs` for `goldset/**/*.csv` → plain blob.
- `tests/test_goldset_sample.py` — **create**: frame, draw, determinism, manifest.
- `tests/test_goldset_packet.py` — **create**: forbidden columns, links, key/packet join.
- `tests/test_goldset_grade.py` — **create**: validation, FN math, weighted, Wilson, projection.

---

## Task 1: Config helpers + package skeleton + frame filter

**Files:**
- Modify: `stratification_scripts/config.py` (append after `get_frozen_snapshot_path`, ~line 363)
- Create: `stratification_scripts/goldset/__init__.py`
- Create: `stratification_scripts/goldset/sample.py`
- Test: `tests/test_goldset_sample.py`

**Interfaces:**
- Consumes: `config.get_project_root()`, `config.get_frozen_snapshot_path(id)`.
- Produces:
  - `config.get_goldset_dir() -> Path`, `config.get_goldset_seed_path(seed_id: str) -> Path`
  - `sample.FRAME_LIFECYCLE: str`, `sample.FRAME_SOURCES: tuple[str, str]`
  - `sample.frame_from_agency_responses(df: pl.DataFrame) -> pl.DataFrame` (pure filter)
  - `sample.load_frame(snapshot_id: str, *, year: int = 2024) -> pl.DataFrame`

- [ ] **Step 1: Write the failing test (config helpers + pure frame filter)**

Create `tests/test_goldset_sample.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_goldset_sample.py -v`
Expected: FAIL — `ModuleNotFoundError: stratification_scripts.goldset` / `AttributeError: config has no attribute 'get_goldset_dir'`.

- [ ] **Step 3: Add config helpers**

In `stratification_scripts/config.py`, immediately after `get_frozen_snapshot_path` (~line 363):

```python
def get_goldset_dir() -> Path:
    """
    Get the repo-root goldset/ directory holding gold-set seed runs.

    Mirrors get_frozen_dir(): a top-level sibling of the re-runnable pipeline,
    holding human-labeled ground-truth artifacts that are committed as plain
    blobs (see the goldset/**/*.csv .gitattributes override).

    Returns:
        Path to goldset/ directory.

    Side Effects:
        Creates the directory if it doesn't exist.
    """
    path = get_project_root() / "goldset"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_goldset_seed_path(seed_id: str) -> Path:
    """
    Get the directory for a specific gold-set seed run.

    Args:
        seed_id: Seed identifier, e.g. "2026-07-17-ce44ac5".

    Returns:
        Path to goldset/<seed_id>/.
    """
    return get_goldset_dir() / seed_id
```

- [ ] **Step 4: Create the package + frame filter**

Create `stratification_scripts/goldset/__init__.py`:

```python
"""Gold-set seed: reproducible stratified sample → blind labeling packet → graded FN rates."""
```

Create `stratification_scripts/goldset/sample.py`:

```python
"""
Gold-set sampling — the reproducible stratified draw from a frozen snapshot.

The frame is the subpopulation where the pipeline's "no response" claim is
*checkable*: a final rule provably exists (FINAL_EFFECTIVE), the pipeline said
"no" (response_found == no), and the answer came from a source a human can
re-derive (web_search or fr_preamble). Within that frame we draw n per source
with a seeded RNG, so the same (seed, snapshot) reproduces the sample forever.

Standalone: NOT imported by cli.py, NOT a pipeline step (like freeze).
"""

from __future__ import annotations

import polars as pl

from stratification_scripts import config

FRAME_LIFECYCLE = "FINAL_EFFECTIVE"
FRAME_RESPONSE_FOUND = "no"
FRAME_SOURCES = ("web_search", "fr_preamble")


def frame_from_agency_responses(df: pl.DataFrame) -> pl.DataFrame:
    """The checkable-"no" frame: FINAL_EFFECTIVE ∧ response_found==no ∧ source∈{web_search, fr_preamble}."""
    return df.filter(
        (pl.col("lifecycle_stage") == FRAME_LIFECYCLE)
        & (pl.col("response_found") == FRAME_RESPONSE_FOUND)
        & (pl.col("response_source").is_in(list(FRAME_SOURCES)))
    )


def load_frame(snapshot_id: str, *, year: int = 2024) -> pl.DataFrame:
    """Read agency_responses_<year>.csv from the pinned snapshot and apply the frame filter."""
    base = config.get_frozen_snapshot_path(snapshot_id)
    src = base / f"makeup/data/agency_responses_{year}.csv"
    df = pl.read_csv(src, infer_schema_length=0)  # all-string, matching the freeze convention
    return frame_from_agency_responses(df)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_goldset_sample.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add stratification_scripts/config.py stratification_scripts/goldset/__init__.py stratification_scripts/goldset/sample.py tests/test_goldset_sample.py
git commit -m "feat(goldset): config helpers + package + checkable-no frame filter"
```

---

## Task 2: Seeded stratified draw + opaque label_row_id + sample manifest

**Files:**
- Modify: `stratification_scripts/goldset/sample.py`
- Test: `tests/test_goldset_sample.py`

**Interfaces:**
- Consumes: `sample.FRAME_SOURCES`, a frame `pl.DataFrame` with columns `comment_id, response_source, response_sample_weight`.
- Produces:
  - `sample.make_label_row_id(seed: int, snapshot_id: str, comment_id: str) -> str`
  - `sample.draw_sample(frame: pl.DataFrame, *, snapshot_id: str, seed: int, n: int = 15, overlap: int = 10) -> pl.DataFrame` — returns sampled rows (all frame columns) plus `label_row_id: str`, `overlap_candidate: bool`, in interleaved (non-stratum-grouped) order.
  - `sample.make_seed_id(snapshot_id: str, moment) -> str` — `"<YYYY-MM-DD>-<snapshot-short>"`.
  - `sample.build_sample_manifest(frame, sampled, *, snapshot_id, seed, n, overlap, moment) -> dict`

- [ ] **Step 1: Write the failing tests (determinism, allocation, opacity, interleave)**

Append to `tests/test_goldset_sample.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_goldset_sample.py -k "draw or label_row or overlap" -v`
Expected: FAIL — `AttributeError: module has no attribute 'draw_sample'`.

- [ ] **Step 3: Implement draw + id + manifest**

Add to the top imports of `stratification_scripts/goldset/sample.py`:

```python
import hashlib
import random
from datetime import datetime, timezone
```

Append to `stratification_scripts/goldset/sample.py`:

```python
def make_label_row_id(seed: int, snapshot_id: str, comment_id: str) -> str:
    """Opaque, stable id for a sampled comment. Encodes no stratum or position."""
    digest = hashlib.sha256(f"{seed}|{snapshot_id}|{comment_id}".encode()).hexdigest()
    return digest[:12]


def draw_sample(
    frame: pl.DataFrame,
    *,
    snapshot_id: str,
    seed: int,
    n: int = 15,
    overlap: int = 10,
) -> pl.DataFrame:
    """Seeded stratified draw of n rows per response_source, interleaved.

    Determinism: within each stratum rows are sorted by comment_id (canonical
    order independent of file/read order), then a single seeded RNG picks the
    sample and shuffles the combined result so strata interleave. Same
    (seed, snapshot_id) ⇒ identical output, forever.
    """
    rng = random.Random(seed)
    picked_frames: list[pl.DataFrame] = []
    for src in FRAME_SOURCES:
        stratum = frame.filter(pl.col("response_source") == src).sort("comment_id")
        if stratum.height < n:
            raise ValueError(
                f"stratum {src!r} has only {stratum.height} rows; cannot draw n={n}"
            )
        idx = sorted(rng.sample(range(stratum.height), n))
        picked_frames.append(stratum[idx])

    combined = pl.concat(picked_frames)
    # Interleave: shuffle a canonical (comment_id-sorted) order with the same RNG.
    order = list(range(combined.height))
    combined = combined.sort("comment_id")
    rng.shuffle(order)
    combined = combined[order]

    label_ids = [
        make_label_row_id(seed, snapshot_id, cid)
        for cid in combined["comment_id"].to_list()
    ]
    if len(set(label_ids)) != len(label_ids):
        raise ValueError("label_row_id collision — widen the id hash")

    overlap_ids = set(rng.sample(label_ids, min(overlap, len(label_ids))))
    return combined.with_columns(
        pl.Series("label_row_id", label_ids),
        pl.Series("overlap_candidate", [lid in overlap_ids for lid in label_ids]),
    )


def _iso_z(moment: datetime) -> str:
    return (
        moment.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def make_seed_id(snapshot_id: str, moment: datetime) -> str:
    """Seed-run directory id: '<YYYY-MM-DD>-<snapshot-short>' (mirrors the freeze id)."""
    snapshot_short = snapshot_id.rsplit("-", 1)[-1]
    return f"{moment.date().isoformat()}-{snapshot_short}"


def _weight_mass(df: pl.DataFrame) -> float:
    return float(df["response_sample_weight"].cast(pl.Float64).sum())


def build_sample_manifest(
    frame: pl.DataFrame,
    sampled: pl.DataFrame,
    *,
    snapshot_id: str,
    seed: int,
    n: int,
    overlap: int,
    moment: datetime,
) -> dict:
    """Provenance for a seed run: the frame shape, per-stratum weight mass, and the sampled rows.

    frame_weight_mass is the Σ weight over ALL rows in each stratum of the frame
    (not just the sampled rows) — the grader's projection denominator.
    """
    strata = {}
    for src in FRAME_SOURCES:
        sfx = frame.filter(pl.col("response_source") == src)
        strata[src] = {
            "frame_rows": sfx.height,
            "frame_weight_mass": round(_weight_mass(sfx), 4),
            "allocated": n,
        }
    sampled_rows = [
        {
            "label_row_id": r["label_row_id"],
            "comment_id": r["comment_id"],
            "response_source": r["response_source"],
            "response_sample_weight": float(r["response_sample_weight"]),
            "overlap_candidate": r["overlap_candidate"],
        }
        for r in sampled.iter_rows(named=True)
    ]
    return {
        "snapshot_id": snapshot_id,
        "created_at": _iso_z(moment),
        "seed": seed,
        "n_per_stratum": n,
        "overlap": overlap,
        "frame_total_rows": frame.height,
        "strata": strata,
        "sampled": sampled_rows,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_goldset_sample.py -v`
Expected: PASS (all sample tests).

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/goldset/sample.py tests/test_goldset_sample.py
git commit -m "feat(goldset): seeded stratified draw, opaque label ids, sample manifest"
```

---

## Task 3: Blind packet + hidden prediction key + link layering

**Files:**
- Create: `stratification_scripts/goldset/packet.py`
- Test: `tests/test_goldset_packet.py`

**Interfaces:**
- Consumes: a `sampled` `pl.DataFrame` from `draw_sample` (columns incl. `comment_id, document_number, agency, rin, response_source, response_sample_weight, label_row_id, overlap_candidate`); the snapshot's `comments_raw` and `federal_register` CSVs.
- Produces:
  - `packet.PACKET_INPUT_COLUMNS: list[str]`, `packet.LABEL_COLUMNS: list[str]`, `packet.FORBIDDEN_IN_PACKET: list[str]`
  - `packet.build_links(sampled: pl.DataFrame) -> pl.DataFrame` — adds `rin_url, nprm_url, final_rule_url, comment_url`.
  - `packet.build_packet_and_key(sampled, *, snapshot_id, year=2024, context=None) -> tuple[pl.DataFrame, pl.DataFrame]`. `context` (optional) injects `(comments_raw_df, fr_df)` for tests; when None, reads them from the pinned snapshot.

- [ ] **Step 1: Write the failing tests (forbidden columns, links, join totality)**

Create `tests/test_goldset_packet.py`:

```python
from __future__ import annotations

import polars as pl

from stratification_scripts.goldset import packet, sample


def _sampled_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "label_row_id": ["a1", "b2", "c3"],
            "comment_id": ["EPA-2024-0001-0001", "FAA-2024-0002-0005", "SBA-2024-0007-0003"],
            "document_number": ["2024-00001", "2024-00002", "2024-00003"],
            "agency": ["EPA", "FAA", "SBA"],
            "rin": ["2060-AV12", "2120-AL55", "3245-AH99"],
            "response_source": ["web_search", "fr_preamble", "web_search"],
            "response_sample_weight": [17.4, 7.5, 12.0],
            "overlap_candidate": [True, False, True],
            # hidden verdicts that must NOT reach the packet:
            "response_found": ["no", "no", "no"],
            "agency_decision": ["", "", ""],
            "response_text": ["secret", "secret", "secret"],
        }
    )


def _context_fixture():
    comments_raw = pl.DataFrame(
        {
            "comment_id": ["EPA-2024-0001-0001", "FAA-2024-0002-0005", "SBA-2024-0007-0003"],
            "comment_text": ["please regulate", "safety concern", "small biz impact"],
            "organization": ["NRDC", "", "Main St LLC"],
            "submitter_type": ["Organization", "Individual", "Organization"],
        }
    )
    fr = pl.DataFrame(
        {
            "document_number": ["2024-00001", "2024-00002", "2024-00003"],
            "title": ["Rule A", "Rule B", "Rule C"],
            "docket_id": ["Docket ID EPA-2024-0001", "FAR Case 2019-015", "Docket ID SBA-2024-0007"],
            "final_action_citation": ["89 FR 102448", "", "89 FR 55000"],
            "final_rule_document_number": ["2024-90001", "", "2024-90003"],
        }
    )
    return comments_raw, fr


def test_links_layer_correctly():
    linked = packet.build_links(_sampled_fixture())
    assert linked["rin_url"].to_list() == [
        "https://www.federalregister.gov/r/2060-AV12",
        "https://www.federalregister.gov/r/2120-AL55",
        "https://www.federalregister.gov/r/3245-AH99",
    ]
    assert linked["nprm_url"][0] == "https://www.federalregister.gov/d/2024-00001"
    assert linked["comment_url"][0] == "https://www.regulations.gov/comment/EPA-2024-0001-0001"


def test_packet_excludes_every_forbidden_column():
    p, _ = packet.build_packet_and_key(
        _sampled_fixture(), snapshot_id="S", context=_context_fixture()
    )
    for col in packet.FORBIDDEN_IN_PACKET:
        assert col not in p.columns, f"forbidden column leaked into packet: {col}"


def test_packet_has_inputs_and_empty_label_columns():
    p, _ = packet.build_packet_and_key(
        _sampled_fixture(), snapshot_id="S", context=_context_fixture()
    )
    assert "comment_text" in p.columns and "rin_url" in p.columns
    for col in packet.LABEL_COLUMNS:
        assert col in p.columns
        assert p[col].fill_null("").to_list() == ["", "", ""]  # empty for labeler to fill


def test_final_rule_url_blank_exactly_when_missing():
    p, _ = packet.build_packet_and_key(
        _sampled_fixture(), snapshot_id="S", context=_context_fixture()
    )
    urls = dict(zip(p["comment_id"].to_list(), p["final_rule_url"].to_list()))
    assert urls["EPA-2024-0001-0001"] == "https://www.federalregister.gov/d/2024-90001"
    assert urls["FAA-2024-0002-0005"] == ""  # final_rule_document_number was blank


def test_no_url_constructed_from_docket_id():
    p, _ = packet.build_packet_and_key(
        _sampled_fixture(), snapshot_id="S", context=_context_fixture()
    )
    # docket_id ships as prose; no column should turn it into a URL
    assert "docket_url" not in p.columns
    joined_urls = " ".join(
        v for c in p.columns if c.endswith("_url") for v in p[c].fill_null("").to_list()
    )
    assert "FAR Case" not in joined_urls and "Docket ID" not in joined_urls


def test_key_and_packet_join_is_total():
    p, key = packet.build_packet_and_key(
        _sampled_fixture(), snapshot_id="S", context=_context_fixture()
    )
    assert sorted(p["label_row_id"].to_list()) == sorted(key["label_row_id"].to_list())
    assert key["label_row_id"].n_unique() == p.height
    # key carries what grading needs, packet does not
    assert "response_source" in key.columns and "response_source" not in p.columns
    assert "response_sample_weight" in key.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_goldset_packet.py -v`
Expected: FAIL — `ModuleNotFoundError: ...goldset.packet`.

- [ ] **Step 3: Implement packet.py**

Create `stratification_scripts/goldset/packet.py`:

```python
"""
The blind labeling packet and its hidden prediction key.

The packet contains ONLY inputs a human needs to judge "did the agency respond
to this comment?" — never the pipeline's own verdict or its answer path. Those
live in prediction_key.csv, joined back only at grade time. A labeler who can
infer the model's answer will unconsciously ratify it, and the gold set stops
being an independent ruler.

Links are layered so every row has a working path to the primary source; the
rin_url fallback is 100% populated. docket_id is prose — it is NEVER a URL.
"""

from __future__ import annotations

import re

import polars as pl

from stratification_scripts import config

# regulations.gov document-id shape, e.g. EPA-HQ-OAR-2024-0001-0001
_REGS_COMMENT_ID = re.compile(r"^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+$")

# Inputs the labeler sees (order = packet column order before the label columns).
PACKET_INPUT_COLUMNS = [
    "label_row_id",
    "comment_id",
    "comment_text",
    "organization",
    "submitter_type",
    "agency",
    "title",
    "rin",
    "document_number",
    "final_rule_document_number",
    "final_action_citation",
    "docket_id",
    "rin_url",
    "nprm_url",
    "final_rule_url",
    "comment_url",
]

# Empty columns the labeler fills in the spreadsheet.
LABEL_COLUMNS = [
    "true_response_found",
    "evidence_quote",
    "evidence_citation",
    "true_agency_decision",
    "labeler_notes",
    "minutes_spent",
    "labeler_id",
]

# The answer path — must never appear in the packet (asserted in tests).
FORBIDDEN_IN_PACKET = [
    "response_found",
    "agency_decision",
    "reasoning",
    "response_text",
    "response_location",
    "response_citation",
    "rtc_document_id",
    "tier2_acceptance_status",
    "tier2_confidence",
    "tier2_text_change_summary",
    "response_source",
]

# What the key carries so grade can join and weight.
KEY_COLUMNS = ["label_row_id", "comment_id", "response_source", "response_sample_weight"]

_FR_BASE = "https://www.federalregister.gov"
_REGS_BASE = "https://www.regulations.gov"


def _url_or_blank(prefix: str, value: str | None) -> str:
    return f"{prefix}{value}" if value not in (None, "") else ""


def build_links(sampled: pl.DataFrame) -> pl.DataFrame:
    """Add layered primary-source links. Never builds a URL from docket_id."""
    has_frdn = "final_rule_document_number" in sampled.columns
    rows = sampled.iter_rows(named=True)
    rin_url, nprm_url, final_url, comment_url = [], [], [], []
    for r in rows:
        rin_url.append(_url_or_blank(f"{_FR_BASE}/r/", r.get("rin")))
        nprm_url.append(_url_or_blank(f"{_FR_BASE}/d/", r.get("document_number")))
        final_url.append(
            _url_or_blank(f"{_FR_BASE}/d/", r.get("final_rule_document_number")) if has_frdn else ""
        )
        cid = r.get("comment_id") or ""
        comment_url.append(
            f"{_REGS_BASE}/comment/{cid}" if _REGS_COMMENT_ID.match(cid) else ""
        )
    return sampled.with_columns(
        pl.Series("rin_url", rin_url),
        pl.Series("nprm_url", nprm_url),
        pl.Series("final_rule_url", final_url),
        pl.Series("comment_url", comment_url),
    )


def _guarded_join(left: pl.DataFrame, right: pl.DataFrame, on: str) -> pl.DataFrame:
    """Left-join after deduping `right` on the key; assert no row fan-out (no many-to-many fan-out)."""
    right1 = right.unique(subset=on, keep="first")
    out = left.join(right1, on=on, how="left")
    if out.height != left.height:
        raise ValueError(f"join on {on!r} fanned out {left.height} -> {out.height} rows")
    return out


def _load_context(snapshot_id: str, year: int):
    base = config.get_frozen_snapshot_path(snapshot_id)
    comments_raw = pl.read_csv(
        base / f"makeup/data/comments_raw_{year}.csv", infer_schema_length=0
    ).select(["comment_id", "comment_text", "organization", "submitter_type"])
    fr = pl.read_csv(
        base / f"output/federal_register_{year}_comments.csv", infer_schema_length=0
    ).select(
        ["document_number", "title", "docket_id", "final_action_citation", "final_rule_document_number"]
    )
    return comments_raw, fr


def build_packet_and_key(
    sampled: pl.DataFrame,
    *,
    snapshot_id: str,
    year: int = 2024,
    context: tuple[pl.DataFrame, pl.DataFrame] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a sampled frame into (blind packet, hidden prediction key).

    context injects (comments_raw, fr) for tests; otherwise they are read from
    the pinned snapshot. Joins are one-to-one guarded (dedupe key + assert no fan-out).
    """
    comments_raw, fr = context if context is not None else _load_context(snapshot_id, year)

    enriched = _guarded_join(sampled, comments_raw, on="comment_id")
    enriched = _guarded_join(enriched, fr, on="document_number")
    enriched = build_links(enriched)

    packet = enriched.select(
        [c for c in PACKET_INPUT_COLUMNS if c in enriched.columns]
    ).with_columns([pl.lit("").alias(c) for c in LABEL_COLUMNS])

    key = enriched.select([c for c in KEY_COLUMNS if c in enriched.columns])
    return packet, key
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_goldset_packet.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/goldset/packet.py tests/test_goldset_packet.py
git commit -m "feat(goldset): blind packet + hidden key + guarded joins + layered links"
```

---

## Task 4: Grader — label loading + validation

**Files:**
- Create: `stratification_scripts/goldset/grade.py`
- Test: `tests/test_goldset_grade.py`

**Interfaces:**
- Consumes: `packet.KEY_COLUMNS`.
- Produces:
  - `grade.VALID_TRUE_FOUND: set[str]`, `grade.VALID_DECISION: set[str]`
  - `grade.load_labels(path) -> pl.DataFrame`
  - `grade.validate_labels(labels: pl.DataFrame, key: pl.DataFrame) -> None` — raises `ValueError` with a clear message on any violation.

- [ ] **Step 1: Write the failing tests (validation rejects bad input)**

Create `tests/test_goldset_grade.py`:

```python
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
    bad = _labels(label_row_id=["a1"], true_response_found=["yes"], evidence_quote=["q"], true_agency_decision=["accept"])
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_goldset_grade.py -v`
Expected: FAIL — `ModuleNotFoundError: ...goldset.grade`.

- [ ] **Step 3: Implement load + validate**

Create `stratification_scripts/goldset/grade.py`:

```python
"""
Grade returned human labels into per-source web-search false-negative rates.

FN rate = P(true == yes | pred == no). Every frame row has pred == no by
construction, so within a stratum the FN rate is just the share the labeler
marked "yes" (a response the pipeline missed). Reported unweighted with a Wilson
CI and as an HT-weighted point estimate, per response_source, with the frame
caveat restated in the output itself.

Validation fails loud: a "yes" without an evidence quote is a guess, not a label.
"""

from __future__ import annotations

import polars as pl

VALID_TRUE_FOUND = {"yes", "no", "uncertain"}
VALID_DECISION = {"accept", "partial", "reject", "uncertain"}


def load_labels(path) -> pl.DataFrame:
    """Read a filled labeling sheet, all columns as strings (empty cells → '')."""
    return pl.read_csv(path, infer_schema_length=0).fill_null("")


def validate_labels(labels: pl.DataFrame, key: pl.DataFrame) -> None:
    """Raise ValueError unless every packet row is labeled exactly once with valid values."""
    label_ids = labels["label_row_id"].to_list()
    key_ids = set(key["label_row_id"].to_list())

    unknown = [i for i in label_ids if i not in key_ids]
    if unknown:
        raise ValueError(f"unknown label_row_id(s) not in the key: {sorted(set(unknown))}")

    missing = key_ids - set(label_ids)
    if missing:
        raise ValueError(f"missing labels for {len(missing)} packet row(s): {sorted(missing)}")

    if len(label_ids) != len(set(label_ids)):
        raise ValueError("duplicate label_row_id(s) in the returned labels")

    for r in labels.iter_rows(named=True):
        tf = (r.get("true_response_found") or "").strip().lower()
        if tf not in VALID_TRUE_FOUND:
            raise ValueError(f"{r['label_row_id']}: true_response_found={tf!r} not in {VALID_TRUE_FOUND}")
        dec = (r.get("true_agency_decision") or "").strip().lower()
        if dec and dec not in VALID_DECISION:
            raise ValueError(f"{r['label_row_id']}: true_agency_decision={dec!r} not in {VALID_DECISION}")
        if tf == "yes" and not (r.get("evidence_quote") or "").strip():
            raise ValueError(f"{r['label_row_id']}: true_response_found=yes requires an evidence_quote")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_goldset_grade.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/goldset/grade.py tests/test_goldset_grade.py
git commit -m "feat(goldset): grader label loading + loud validation"
```

---

## Task 5: Grader — FN statistics, Wilson CI, HT-weighted projection

**Files:**
- Modify: `stratification_scripts/goldset/grade.py`
- Test: `tests/test_goldset_grade.py`

**Interfaces:**
- Consumes: validated `labels`, `key`, and the sample `manifest` dict (for per-stratum `frame_weight_mass`).
- Produces:
  - `grade.wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]`
  - `grade.compute_stats(labels: pl.DataFrame, key: pl.DataFrame, manifest: dict) -> dict`

- [ ] **Step 1: Write the failing tests (Wilson bounds, FN math, weighting, projection)**

Append to `tests/test_goldset_grade.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_goldset_grade.py -k "wilson or fn_rate or weighted or contrast" -v`
Expected: FAIL — `AttributeError: module has no attribute 'wilson_ci'`.

- [ ] **Step 3: Implement stats**

Add to the imports of `stratification_scripts/goldset/grade.py`:

```python
import math

from stratification_scripts.goldset.sample import FRAME_SOURCES
```

Append to `stratification_scripts/goldset/grade.py`:

```python
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n. n==0 → (0.0, 1.0)."""
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _stratum_stats(rows: list[dict], frame_weight_mass: float) -> dict:
    """rows: joined (label ⋈ key) dicts for one response_source stratum."""
    n = len(rows)
    is_yes = [(r["true_response_found"].strip().lower() == "yes") for r in rows]
    yes = sum(is_yes)
    uncertain = sum(1 for r in rows if r["true_response_found"].strip().lower() == "uncertain")
    weights = [float(r["response_sample_weight"]) for r in rows]

    fn_unweighted = yes / n if n else 0.0
    lo, hi = wilson_ci(yes, n)

    wsum = sum(weights)
    wyes = sum(w for w, y in zip(weights, is_yes) if y)
    fn_weighted = (wyes / wsum) if wsum else 0.0

    return {
        "n": n,
        "yes": yes,
        "uncertain": uncertain,
        "fn_unweighted": fn_unweighted,
        "fn_unweighted_ci95": [lo, hi],
        "fn_weighted": fn_weighted,
        "frame_weight_mass": frame_weight_mass,
        "projected_missed": fn_weighted * frame_weight_mass,
    }


def compute_stats(labels: pl.DataFrame, key: pl.DataFrame, manifest: dict) -> dict:
    """Per-source FN stats + the web-vs-fr contrast. Assumes labels already validated.

    FN denominator is every sampled row in the stratum (all pred==no); `uncertain`
    labels stay in the denominator (a conservative FN estimate) and are also
    reported explicitly so the reader can see how many were unresolved.
    """
    joined = key.join(labels, on="label_row_id", how="left")
    per_source = {}
    for src in FRAME_SOURCES:
        rows = [r for r in joined.iter_rows(named=True) if r["response_source"] == src]
        mass = float(manifest["strata"].get(src, {}).get("frame_weight_mass", 0.0))
        per_source[src] = _stratum_stats(rows, mass)

    contrast = per_source["web_search"]["fn_unweighted"] - per_source["fr_preamble"]["fn_unweighted"]
    return {"strata": per_source, "contrast_web_minus_fr": contrast}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_goldset_grade.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/goldset/grade.py tests/test_goldset_grade.py
git commit -m "feat(goldset): FN stats, Wilson CI, HT-weighted projection, contrast"
```

---

## Task 6: Grader — report render + results writers

**Files:**
- Modify: `stratification_scripts/goldset/grade.py`
- Test: `tests/test_goldset_grade.py`

**Interfaces:**
- Consumes: a `stats` dict from `compute_stats`.
- Produces:
  - `grade.render_report(stats: dict, *, n_per_stratum: int) -> str` — markdown with the honesty caveats embedded.
  - `grade.write_results(stats: dict, seed_dir, *, n_per_stratum: int) -> None` — writes `results.json` + `results.md`.

- [ ] **Step 1: Write the failing tests (caveats appear in the rendered output)**

Append to `tests/test_goldset_grade.py`:

```python
def test_report_embeds_honesty_caveats():
    stats = {
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
    md = grade.render_report(stats, n_per_stratum=15)
    assert "directional" in md.lower()  # not-publishable caveat
    assert "final rule provably exists" in md.lower()  # frame caveat
    assert "web_search" in md and "fr_preamble" in md


def test_write_results_creates_both_files(tmp_path):
    stats = {
        "strata": {
            "web_search": {"n": 15, "yes": 4, "uncertain": 1, "fn_unweighted": 0.27,
                           "fn_unweighted_ci95": [0.1, 0.5], "fn_weighted": 0.30,
                           "frame_weight_mass": 2615.0, "projected_missed": 784.5},
            "fr_preamble": {"n": 15, "yes": 2, "uncertain": 0, "fn_unweighted": 0.13,
                            "fn_unweighted_ci95": [0.03, 0.4], "fn_weighted": 0.13,
                            "frame_weight_mass": 1719.0, "projected_missed": 223.5},
        },
        "contrast_web_minus_fr": 0.14,
    }
    grade.write_results(stats, tmp_path, n_per_stratum=15)
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "results.md").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_goldset_grade.py -k "report or write_results" -v`
Expected: FAIL — `AttributeError: module has no attribute 'render_report'`.

- [ ] **Step 3: Implement report + writers**

Add to the imports of `stratification_scripts/goldset/grade.py`:

```python
import json
from pathlib import Path
```

Append to `stratification_scripts/goldset/grade.py`:

```python
def render_report(stats: dict, *, n_per_stratum: int) -> str:
    """A short human-facing table with the honesty caveats emitted inline (not just in docs)."""
    lines = ["# Gold-set seed — web-search false-negative rates", ""]
    lines.append(
        f"n = {n_per_stratum}/stratum. Results are **directional, not publishable** "
        f"(Wilson CI half-width ≈ ±25pp at this size)."
    )
    lines.append(
        "Scope: the **\"final rule provably exists\"** subpopulation (FINAL_EFFECTIVE ∧ "
        "checkable \"no\"), **not** the 26,159 population — and not the NO_RIN mass "
        "(72% of web-search \"no\"s) where underestimation is most suspected."
    )
    lines.append("")
    lines.append("| source | n | yes (missed) | uncertain | FN unweighted | 95% CI | FN weighted | projected missed |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for src, s in stats["strata"].items():
        lo, hi = s["fn_unweighted_ci95"]
        lines.append(
            f"| {src} | {s['n']} | {s['yes']} | {s['uncertain']} | "
            f"{s['fn_unweighted']:.1%} | [{lo:.1%}, {hi:.1%}] | {s['fn_weighted']:.1%} | "
            f"{s['projected_missed']:.0f} |"
        )
    lines.append("")
    lines.append(
        f"**Contrast (web_search − fr_preamble, unweighted):** "
        f"{stats['contrast_web_minus_fr']:+.1%} — the claim under test (is grounded's FN lower?)."
    )
    lines.append("")
    lines.append(
        "The weighted estimate ships **without** a CI: a design-based variance estimator "
        "for a weighted proportion is not credible at n=15. It is a point estimate, stated as such."
    )
    return "\n".join(lines) + "\n"


def write_results(stats: dict, seed_dir, *, n_per_stratum: int) -> None:
    """Write results.json (machine) + results.md (humans) into the seed run directory."""
    seed_dir = Path(seed_dir)
    (seed_dir / "results.json").write_text(json.dumps(stats, indent=2) + "\n")
    (seed_dir / "results.md").write_text(render_report(stats, n_per_stratum=n_per_stratum))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_goldset_grade.py -v`
Expected: PASS (all grade tests).

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/goldset/grade.py tests/test_goldset_grade.py
git commit -m "feat(goldset): results report with inline honesty caveats + writers"
```

---

## Task 7: CLI, module entrypoint, .gitattributes override, real-data smoke

**Files:**
- Create: `stratification_scripts/goldset/cli.py`
- Create: `stratification_scripts/goldset/__main__.py`
- Modify: `.gitattributes`
- Test: `tests/test_goldset_sample.py` (append the overwrite-guard test)

**Interfaces:**
- Consumes: `sample.*`, `packet.build_packet_and_key`, `grade.*`, `config.get_goldset_seed_path`.
- Produces:
  - `cli.cmd_sample(args) -> int`, `cli.cmd_grade(args) -> int`, `cli.main(argv=None) -> int`
  - `python -m stratification_scripts.goldset sample|grade ...`

- [ ] **Step 1: Write the failing test (sample refuses to overwrite an existing seed dir)**

Append to `tests/test_goldset_sample.py`:

```python
from stratification_scripts.goldset import cli


def test_sample_refuses_to_overwrite_existing_seed_dir(tmp_path, monkeypatch):
    seed_dir = tmp_path / "2026-07-17-ce44ac5"
    seed_dir.mkdir(parents=True)
    monkeypatch.setattr(cli.config, "get_goldset_seed_path", lambda sid: seed_dir)
    with pytest.raises(FileExistsError, match="already exists"):
        cli.write_seed_run(
            packet=pl.DataFrame({"label_row_id": ["a"]}),
            key=pl.DataFrame({"label_row_id": ["a"]}),
            manifest={"seed_id": "2026-07-17-ce44ac5"},
            seed_id="2026-07-17-ce44ac5",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_goldset_sample.py -k overwrite -v`
Expected: FAIL — `ModuleNotFoundError: ...goldset.cli` / no `write_seed_run`.

- [ ] **Step 3: Implement cli.py + __main__.py**

Create `stratification_scripts/goldset/cli.py`:

```python
"""
Gold-set CLI — standalone, never wired into the pipeline cli.py.

    python -m stratification_scripts.goldset sample --snapshot <id> [--n 15] [--seed 0] [--overlap 10]
    python -m stratification_scripts.goldset grade <seed-id> --labels <path>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from stratification_scripts import config
from stratification_scripts.goldset import grade, packet, sample


def write_seed_run(*, packet: pl.DataFrame, key: pl.DataFrame, manifest: dict, seed_id: str) -> Path:
    """Write manifest + packet + key into goldset/<seed_id>/; refuse to overwrite."""
    seed_dir = config.get_goldset_seed_path(seed_id)
    if seed_dir.exists():
        raise FileExistsError(f"seed run already exists: {seed_dir}")
    seed_dir.mkdir(parents=True)
    (seed_dir / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    packet.write_csv(seed_dir / "labeling_packet.csv")
    key.write_csv(seed_dir / "prediction_key.csv")
    return seed_dir


def cmd_sample(args) -> int:
    frame = sample.load_frame(args.snapshot, year=args.year)
    sampled = sample.draw_sample(
        frame, snapshot_id=args.snapshot, seed=args.seed, n=args.n, overlap=args.overlap
    )
    moment = datetime.now(timezone.utc)
    seed_id = sample.make_seed_id(args.snapshot, moment)
    manifest = sample.build_sample_manifest(
        frame, sampled, snapshot_id=args.snapshot, seed=args.seed,
        n=args.n, overlap=args.overlap, moment=moment,
    )
    manifest["seed_id"] = seed_id
    pkt, key = packet.build_packet_and_key(sampled, snapshot_id=args.snapshot, year=args.year)
    seed_dir = write_seed_run(packet=pkt, key=key, manifest=manifest, seed_id=seed_id)
    print(f"wrote seed run {seed_id} ({pkt.height} rows) → {seed_dir}")
    print("next: label labeling_packet.csv in a spreadsheet, save as labels_returned.csv, then `grade`.")
    return 0


def cmd_grade(args) -> int:
    seed_dir = config.get_goldset_seed_path(args.seed_id)
    manifest = json.loads((seed_dir / "sample_manifest.json").read_text())
    key = pl.read_csv(seed_dir / "prediction_key.csv", infer_schema_length=0)
    labels = grade.load_labels(args.labels)
    grade.validate_labels(labels, key)
    stats = grade.compute_stats(labels, key, manifest)
    grade.write_results(stats, seed_dir, n_per_stratum=manifest.get("n_per_stratum", args.n))
    print((seed_dir / "results.md").read_text())
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="goldset",
        description="Draw a blind gold-set labeling packet from a frozen snapshot, then grade returned labels.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample", help="Draw a stratified sample and write the blind packet.")
    p_sample.add_argument("--snapshot", required=True, help="Frozen snapshot id, e.g. 2026-07-15-ce44ac5.")
    p_sample.add_argument("--year", type=int, default=2024)
    p_sample.add_argument("--n", type=int, default=15, help="Rows per stratum.")
    p_sample.add_argument("--seed", type=int, default=0, help="RNG seed (reproducibility).")
    p_sample.add_argument("--overlap", type=int, default=10, help="Rows flagged for double-labeling.")

    p_grade = sub.add_parser("grade", help="Grade returned labels for a seed run.")
    p_grade.add_argument("seed_id", help="Seed run id (the goldset/<seed-id> dir name).")
    p_grade.add_argument("--labels", required=True, help="Path to the filled labels_returned.csv.")
    p_grade.add_argument("--n", type=int, default=15, help="Fallback n if absent from the manifest.")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "sample":
            return cmd_sample(args)
        if args.cmd == "grade":
            return cmd_grade(args)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1  # unreachable: subparser required


if __name__ == "__main__":
    sys.exit(main())
```

Create `stratification_scripts/goldset/__main__.py`:

```python
import sys

from stratification_scripts.goldset.cli import main

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the overwrite-guard test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_goldset_sample.py -k overwrite -v`
Expected: PASS.

- [ ] **Step 5: Add the .gitattributes override + verify with git check-attr**

Append to `.gitattributes`:

```
goldset/**/*.csv -filter -diff -merge text
```

Then verify the override actually wins over the blanket `*.csv filter=lfs` rule:

Run:
```bash
git check-attr filter diff merge -- goldset/2026-07-17-x/labeling_packet.csv
git check-attr filter -- frozen/2026-07-15-ce44ac5/makeup/data/agency_responses_2024.csv
```
Expected: the `goldset/...` path reports `filter: unset`, `diff: unset`, `merge: unset` (plain blob); the `frozen/...` path still reports `filter: lfs` (blanket rule intact elsewhere).

- [ ] **Step 6: Full test suite green**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS — all goldset tests plus the pre-existing suite unaffected.

- [ ] **Step 7: Real-data smoke — draw an actual packet against the frozen snapshot**

Run:
```bash
.venv/bin/python -m stratification_scripts.goldset sample --snapshot 2026-07-15-ce44ac5 --seed 0
```
Expected: prints `wrote seed run <today>-ce44ac5 (30 rows) → .../goldset/<today>-ce44ac5`.

Then verify the artifacts by inspection:
```bash
.venv/bin/python -c "
import polars as pl, json, glob
d = sorted(glob.glob('goldset/*-ce44ac5'))[-1]
p = pl.read_csv(d + '/labeling_packet.csv', infer_schema_length=0)
k = pl.read_csv(d + '/prediction_key.csv', infer_schema_length=0)
m = json.load(open(d + '/sample_manifest.json'))
forbidden = {'response_found','agency_decision','reasoning','response_text','response_location','response_citation','rtc_document_id','tier2_acceptance_status','tier2_confidence','tier2_text_change_summary','response_source'}
assert p.height == 30, p.height
assert not (forbidden & set(p.columns)), forbidden & set(p.columns)
assert set(p['label_row_id']) == set(k['label_row_id'])
assert all(u.startswith('https://www.federalregister.gov/r/') for u in p['rin_url']), 'rin_url must be 100%'
assert m['strata']['web_search']['frame_rows'] == 150 and m['strata']['fr_preamble']['frame_rows'] == 228
print('SMOKE OK:', d, '| 30 rows, no leak, rin_url 100%, frame 150/228')
"
```
Expected: `SMOKE OK: ...` — 30 rows, no forbidden columns, key/packet aligned, rin_url universal, frame counts match the verified 150/228.

- [ ] **Step 8: Commit (code + the real seed run as plain blobs)**

```bash
git add stratification_scripts/goldset/cli.py stratification_scripts/goldset/__main__.py .gitattributes tests/test_goldset_sample.py
git add goldset/
git status --short  # confirm goldset/*.csv are staged as plain text, NOT lfs pointers
git commit -m "feat(goldset): argparse CLI (sample/grade) + entrypoint + plain-blob gitattributes + first seed run"
```

Note: the labeled `labels_returned.csv` and `results.*` are produced later (after Jonathan hand-labels) via `grade`; they commit then.

---

## Self-Review

**1. Spec coverage** (each spec section → task):

| Spec section | Covered by |
|---|---|
| Frame (FINAL_EFFECTIVE ∧ no ∧ {web_search,fr_preamble}) | Task 1 (`frame_from_agency_responses`, verified 378 rows) |
| Strata & seeded sampling, overlap flag | Task 2 (`draw_sample`, determinism + interleave + overlap tests) |
| Weights (per-row + cell mass) | Task 2 manifest `frame_weight_mass`; Task 5 weighted estimate + projection |
| Instrument = spreadsheet | Task 3/7 (CSV packet, no UI) |
| Blind packet + hidden key + anti-anchoring | Task 3 (`FORBIDDEN_IN_PACKET`, opaque id, interleave) |
| Links layered; no docket_id URL | Task 3 (`build_links`, `test_no_url_constructed_from_docket_id`) |
| Grader validation | Task 4 |
| Grader stats (FN, Wilson, weighted, contrast, projection) | Task 5 |
| Honesty caveats in output | Task 6 (`render_report`) |
| Structure/artifacts + CLI | Task 7 |
| `.gitattributes` plain-blob override (verified with check-attr) | Task 7 Step 5 |
| Testing (12 cases) | mapped: T1→#1, T2→#2/3/5/6, T3→#4/7/8, T5→#10, T6→#9/11, T7→#12 |

Frame narrowness / NO_RIN exclusion / blindness-is-trust: these are stated caveats, discharged by the report text (Task 6), not code — correct.

**2. Placeholder scan:** none — every step carries real code and exact commands.

**3. Type consistency:** `label_row_id` (str) threads sample→packet(KEY_COLUMNS)→grade(join key) consistently; `response_sample_weight` cast to float only inside stats/manifest; `frame_weight_mass` produced in Task 2 manifest and consumed by name in Task 5. `FRAME_SOURCES` defined once in `sample` and imported by `grade`. Names align.

---

## Execution Handoff

Plan complete. Because the anti-anchoring guarantees (forbidden-column denylist, opaque ids, guarded joins) are load-bearing and worth a fresh gate per task, **subagent-driven-development** (fresh subagent per task + two-stage review) is the right execution mode here.
