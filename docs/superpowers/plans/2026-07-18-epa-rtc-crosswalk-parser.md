# EPA RTC Crosswalk Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse the EPA CCL5 Response-to-Comments PDF into a structured per-comment crosswalk (Document ID → topic(s) → disposition text), plus Exhibit 2 commenters and per-topic Agency Topic Discussions.

**Architecture:** Standalone `stratification_scripts/rtc_parser/` module mirroring `goldset`/`freeze`. One io/fitz unit (`extract`) feeds pure string-processing units (`clean` → `exhibit2`/`responses`/`topics` → `crosswalk`). Deterministic, rule-based, no LLM. Never imported by the pipeline `cli.py`.

**Tech Stack:** Python 3.12, PyMuPDF (`fitz`) for PDF→text, polars for CSV writing (already the repo's tabular lib), stdlib `re`/`dataclasses`/`json`. pytest.

**Spec:** `docs/superpowers/specs/2026-07-18-epa-rtc-crosswalk-parser-design.md`

## Global Constraints

- **No pipeline wiring.** Never imported by `stratification_scripts/cli.py`. Invoked only via `python -m stratification_scripts.rtc_parser`.
- **No validation claims / no LLM.** Deterministic parse. Tests assert unit behavior + end-to-end structure, never correctness-against-ground-truth.
- **Docket prefix parameterized**, default `EPA-HQ-OW-2018-0594`. Document-ID shape: `<prefix>-\d{4}`.
- **Fidelity rule:** topic cross-refs preserve `raw`, carry `canonical`+`resolved`; never dropped, never force-matched.
- **PDF not committed** — gitignored under `rtc/inputs/`; recorded by sha256 + source URL in `parse_manifest.json`.
- **Frozen goldset run untouched.** This module writes only under `rtc/`.
- Commit messages end with the `Co-Authored-By: Claude Opus 4.8 (1M context)` trailer.

---

## File Structure

```
stratification_scripts/rtc_parser/
  __init__.py          # docstring + public exports
  __main__.py          # sys.exit(cli.main())
  cli.py               # argparse `parse` command; orchestrates + writes outputs
  models.py            # Commenter, TopicRef, CommentRecord dataclasses
  extract.py           # extract_pages(pdf)->list[str]; pdf_sha256(pdf)->str   [only fitz/io]
  clean.py             # strip_page_headers(pages)->str; strip_running_headers(text)->str
  exhibit2.py          # parse_commenters(text, docket_prefix)->list[Commenter]
  responses.py         # split_comment_blocks(text)->list[CommentBlock]
  topics.py            # CANONICAL_TOPICS; split_topic_discussions(text)->dict; resolve_refs(raw)->list[TopicRef]
  crosswalk.py         # assemble(commenters, blocks, discussions)->list[CommentRecord]
config.py              # + get_rtc_dir(), get_rtc_output_path(slug), get_rtc_inputs_dir()
tests/
  test_rtc_extract.py
  test_rtc_clean.py
  test_rtc_exhibit2.py
  test_rtc_topics.py
  test_rtc_responses.py
  test_rtc_crosswalk.py
  test_rtc_cli.py
  fixtures/rtc_ccl5_slice.txt   # committed real-doc text slice
```

---

### Task 1: Package scaffold + config paths + gitignore

**Files:**
- Create: `stratification_scripts/rtc_parser/__init__.py`, `__main__.py`
- Modify: `stratification_scripts/config.py` (add rtc path helpers near `get_goldset_dir`)
- Modify: `.gitignore` (ignore `rtc/inputs/`)
- Test: `tests/test_rtc_cli.py` (config-path test to start)

**Interfaces:**
- Produces: `config.get_rtc_dir() -> Path` (`<project_root>/rtc`); `config.get_rtc_output_path(slug) -> Path` (`rtc/<slug>`); `config.get_rtc_inputs_dir() -> Path` (`rtc/inputs`).

- [ ] **Step 1: Failing test**
```python
# tests/test_rtc_cli.py
from stratification_scripts import config

def test_rtc_paths_are_project_siblings():
    assert config.get_rtc_dir() == config.get_project_root() / "rtc"
    assert config.get_rtc_output_path("ccl5") == config.get_project_root() / "rtc" / "ccl5"
    assert config.get_rtc_inputs_dir() == config.get_project_root() / "rtc" / "inputs"
```
- [ ] **Step 2:** Run `pytest tests/test_rtc_cli.py -v` → FAIL (AttributeError).
- [ ] **Step 3:** Add helpers to `config.py` mirroring `get_goldset_dir`/`get_goldset_seed_path`:
```python
def get_rtc_dir() -> Path:
    return get_project_root() / "rtc"

def get_rtc_output_path(slug: str) -> Path:
    return get_rtc_dir() / slug

def get_rtc_inputs_dir() -> Path:
    return get_rtc_dir() / "inputs"
```
  Create `__init__.py` (module docstring) and `__main__.py` (`import sys; from .cli import main; sys.exit(main())`). Add `rtc/inputs/` to `.gitignore`.
- [ ] **Step 4:** Run test → PASS.
- [ ] **Step 5:** Commit `feat(rtc): package scaffold + config paths + gitignore inputs`.

---

### Task 2: models.py

**Files:** Create `stratification_scripts/rtc_parser/models.py`; Test `tests/test_rtc_crosswalk.py` (start).

**Interfaces — Produces:**
```python
@dataclass(frozen=True)
class Commenter:
    number: int; document_id: str; first_name: str; last_name: str; organization: str

@dataclass(frozen=True)
class TopicRef:
    raw: str; canonical: str | None; resolved: bool

@dataclass
class CommentRecord:
    commenter_number: int
    document_id: str | None
    first_name: str; last_name: str; organization: str
    comment_excerpt: str
    has_individual_response: bool
    topic_refs: list[TopicRef]
    individual_response_supplemental: str
    topic_discussions: dict[str, str]  # canonical -> disposition text
```

- [ ] **Step 1: Failing test**
```python
# tests/test_rtc_crosswalk.py
from stratification_scripts.rtc_parser.models import Commenter, TopicRef, CommentRecord

def test_models_construct():
    c = Commenter(54, "EPA-HQ-OW-2018-0594-0054", "", "", "Anonymous")
    t = TopicRef(raw="PFAS", canonical="Per- and Polyfluoroalkyl substances (PFAS)", resolved=True)
    r = CommentRecord(54, c.document_id, "", "", "Anonymous", "excerpt", True, [t], "", {})
    assert c.number == 54 and t.resolved and r.topic_refs[0].raw == "PFAS"
```
- [ ] **Step 2:** Run → FAIL (module missing).
- [ ] **Step 3:** Implement `models.py` with the dataclasses above (`from __future__ import annotations`).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(rtc): record dataclasses`.

---

### Task 3: extract.py (PDF → per-page text, sha256)

**Files:** Create `stratification_scripts/rtc_parser/extract.py`; Test `tests/test_rtc_extract.py`.

**Interfaces — Produces:** `extract_pages(pdf_path: Path) -> list[str]`; `pdf_sha256(pdf_path: Path) -> str`.

- [ ] **Step 1: Failing test** (generate a tiny PDF in-test via fitz — deterministic, no real doc needed; plus a skipif real-doc page count):
```python
# tests/test_rtc_extract.py
import hashlib
from pathlib import Path
import fitz
import pytest
from stratification_scripts.rtc_parser import extract

def _make_pdf(tmp_path: Path) -> Path:
    doc = fitz.open()
    for txt in ["PAGE ONE hello", "PAGE TWO world"]:
        page = doc.new_page()
        page.insert_text((72, 72), txt)
    p = tmp_path / "tiny.pdf"
    doc.save(p); doc.close()
    return p

def test_extract_pages_returns_text_per_page(tmp_path):
    pages = extract.extract_pages(_make_pdf(tmp_path))
    assert len(pages) == 2
    assert "hello" in pages[0] and "world" in pages[1]

def test_pdf_sha256_matches_hashlib(tmp_path):
    p = _make_pdf(tmp_path)
    assert extract.pdf_sha256(p) == hashlib.sha256(p.read_bytes()).hexdigest()
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement:
```python
from __future__ import annotations
import hashlib
from pathlib import Path

def extract_pages(pdf_path: Path) -> list[str]:
    import fitz
    doc = fitz.open(pdf_path)
    try:
        return [doc[i].get_text() for i in range(doc.page_count)]
    finally:
        doc.close()

def pdf_sha256(pdf_path: Path) -> str:
    return hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
```
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(rtc): fitz page extraction + pdf sha256`.

---

### Task 4: clean.py (strip page noise + running headers)

**Files:** Create `stratification_scripts/rtc_parser/clean.py`; Test `tests/test_rtc_clean.py`.

**Interfaces — Produces:** `strip_page_headers(pages: list[str]) -> str` (joins pages, drops the 5 EPA/page-number noise lines); `strip_running_headers(text: str) -> str` (drops `Agency Discussion on …` / `Comments Received on …` lines — for body captures only).

- [ ] **Step 1: Failing test**
```python
# tests/test_rtc_clean.py
from stratification_scripts.rtc_parser import clean

PAGE = ("EPA-OGWDW\nDraft CCL 5 Response to Comments\nEPA 815-R-22-001\n"
        "October 2022\nPage 14 of 159\n\nReal body line\n")

def test_strip_page_headers_removes_noise_keeps_body():
    out = clean.strip_page_headers([PAGE, PAGE])
    assert "Real body line" in out
    for noise in ["EPA-OGWDW", "EPA 815-R-22-001", "Page 14 of 159", "October 2022"]:
        assert noise not in out

def test_strip_running_headers_removes_section_headers():
    txt = "Agency Discussion on General Comments\nkept\nComments Received on PFAS\nalso kept\n"
    out = clean.strip_running_headers(txt)
    assert "kept" in out and "also kept" in out
    assert "Agency Discussion on" not in out and "Comments Received on" not in out
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement:
```python
from __future__ import annotations
import re

_NOISE = re.compile(
    r"^(?:EPA-OGWDW|Draft CCL 5 Response to Comments|EPA 815-R-22-001|"
    r"October 2022|Page \d+ of \d+)\s*$"
)
_RUNNING = re.compile(r"^(?:Agency Discussion on|Comments Received on) .+$")

def strip_page_headers(pages: list[str]) -> str:
    lines = []
    for page in pages:
        for line in page.splitlines():
            if not _NOISE.match(line.strip()):
                lines.append(line)
    return "\n".join(lines)

def strip_running_headers(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not _RUNNING.match(l.strip()))
```
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(rtc): page-noise + running-header cleaners`.

---

### Task 5: exhibit2.py (commenter table)

**Files:** Create `stratification_scripts/rtc_parser/exhibit2.py`; Test `tests/test_rtc_exhibit2.py`.

**Interfaces — Consumes:** `Commenter`. **Produces:** `parse_commenters(text: str, docket_prefix: str = "EPA-HQ-OW-2018-0594") -> list[Commenter]`.

**Approach:** Restrict to the Exhibit 2 region (from the `Exhibit 2: List of Public Commenters` header to the `2. Comments and EPA Responses by Topic` marker) so stray Document IDs in prose aren't captured. Within the region, walk lines: a line that is a bare integer immediately followed (allowing blank lines) by a `<prefix>-\d{4}` line starts a row; the next up-to-3 non-empty lines before the next integer/doc-id are first/last/org, with org being the remainder (may wrap). Anonymous rows: first/last blank.

- [ ] **Step 1: Failing test**
```python
# tests/test_rtc_exhibit2.py
from stratification_scripts.rtc_parser import exhibit2

REGION = """Exhibit 2: List of Public Commenters
Comment Information
Submitter Information
Commenter
Number
Document ID
First Name
Last Name
Organization Name
54
EPA-HQ-OW-2018-0594-0054



Anonymous
56
EPA-HQ-OW-2018-0594-0056
Brian
Callahan
Private Citizen
70
EPA-HQ-OW-2018-0594-0070


National Ground
Water Association
(NGWA)
2. Comments and EPA Responses by Topic
"""

def test_parses_rows_including_anonymous_and_wrapped_org():
    rows = {c.number: c for c in exhibit2.parse_commenters(REGION)}
    assert rows[54].organization == "Anonymous" and rows[54].first_name == ""
    assert rows[56].first_name == "Brian" and rows[56].last_name == "Callahan"
    assert rows[56].document_id == "EPA-HQ-OW-2018-0594-0056"
    assert rows[70].organization == "National Ground Water Association (NGWA)"
    assert rows[70].first_name == "" and rows[70].last_name == ""
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement a region-bounded line walker. Algorithm: slice text between `Exhibit 2: List of Public Commenters` and `2. Comments and EPA Responses by Topic`; iterate lines; when a bare-int line is followed by a doc-id line, open a row `{number, document_id}`, then collect subsequent non-empty lines until the next bare-int-then-doc-id boundary; map collected fields — if exactly the header labels are skipped, treat: names may be blank. Since first/last/org positions vary with blanks, use the rule: **the last collected line(s) that are not plausible person-names form the org**; simpler robust rule chosen for the prototype — collect the block's non-empty lines `L`; the org is the *trailing* run that continues an organization (join all lines after the (optional) first two name lines). Concretely: if the block after doc-id has ≥1 line, and the row is Anonymous/Private-Citizen style (single trailing token), first=last=""; else first=L[0], last=L[1], org=" ".join(L[2:]). Handle the "names blank → org only" case by detecting when only 1–N org lines exist with no separate first/last (heuristic: if the first two lines look like an org continuation). *Decision Ledger #A: the first/last/org disambiguation heuristic — see Ledger below; unit-tested on the three canonical shapes (named person, anonymous, org-only wrapped).* 
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(rtc): Exhibit 2 commenter-table parser`.

---

### Task 6: topics.py — canonical topics + discussion splitter

**Files:** Create `stratification_scripts/rtc_parser/topics.py`; Test `tests/test_rtc_topics.py`.

**Interfaces — Produces:** `CANONICAL_TOPICS: list[str]` (the 22 running-header names, verbatim); `split_topic_discussions(text: str) -> dict[str, str]` (canonical topic → disposition text).

**Approach:** For each `Agency Topic Discussion:` occurrence, topic = nearest preceding `Agency Discussion on (?P<t>.+)` header; body = text after `Agency Topic Discussion:` up to the first `Comment Excerpt from Commenter` OR next `Agency Discussion on`; run `strip_running_headers` on the body.

- [ ] **Step 1: Failing test**
```python
# tests/test_rtc_topics.py
from stratification_scripts.rtc_parser import topics

TWO_TOPICS = """Agency Discussion on General Comments
Agency Topic Discussion:
EPA received many general comments. The Agency agrees.
Comments Received on General Comments
Comment Excerpt from Commenter 52
body
Agency Discussion on Length of CCL 5
Agency Topic Discussion:
EPA disagrees the list is too long.
Comment Excerpt from Commenter 71
body
"""

def test_split_topic_discussions_keys_and_text():
    d = topics.split_topic_discussions(TWO_TOPICS)
    assert set(d) == {"General Comments", "Length of CCL 5"}
    assert "The Agency agrees." in d["General Comments"]
    assert "too long" in d["Length of CCL 5"]
    assert "Comment Excerpt" not in d["General Comments"]
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `CANONICAL_TOPICS` (the 22 names from the spec) and `split_topic_discussions` using `re.finditer` on `Agency Topic Discussion:` with the preceding-header lookup and the body end-anchor described above; strip running headers from each body; `.strip()`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(rtc): topic discussion splitter + canonical topics`.

---

### Task 7: topics.py — cross-ref resolution

**Files:** Modify `stratification_scripts/rtc_parser/topics.py`; Test `tests/test_rtc_topics.py`.

**Interfaces — Consumes:** `TopicRef`. **Produces:** `resolve_refs(raw_clause: str) -> list[TopicRef]`.

**Resolution order** (per mention, after normalizing whitespace/newlines to single spaces and splitting on `,` and ` and `): (1) exact case-insensitive vs canonical; (2) alias table (`{"pfas": <canonical>, "dbps": <canonical>}` — high-confidence expansions only); (3) unique substring (mention appears in exactly one canonical); else unresolved (`canonical=None, resolved=False`). Never force-match; never drop.

- [ ] **Step 1: Failing test**
```python
def test_resolve_refs_abbrev_multi_and_unresolved():
    refs = topics.resolve_refs("General Comments, PFAS, and 1,4-Dioxane")
    by_raw = {r.raw: r for r in refs}
    assert by_raw["General Comments"].canonical == "General Comments"
    assert by_raw["PFAS"].canonical == "Per- and Polyfluoroalkyl substances (PFAS)"
    assert by_raw["1,4-Dioxane"].resolved is False and by_raw["1,4-Dioxane"].canonical is None
    assert len(refs) == 3  # nothing dropped

def test_resolve_refs_joins_linebreak_split_topic():
    refs = topics.resolve_refs("Draft CCL\n5-Microbes")
    assert refs[0].canonical == "Draft CCL 5-Microbes" and refs[0].resolved
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `_split_mentions` (normalize `\s+`→` `, then split on `,`/` and `, drop empties/strip), a lowercase canonical index, the alias map, and `resolve_refs` applying the order above.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(rtc): cross-ref topic resolution (raw+canonical+flag)`.

---

### Task 8: responses.py (comment blocks)

**Files:** Create `stratification_scripts/rtc_parser/responses.py`; Test `tests/test_rtc_responses.py`.

**Interfaces — Produces:**
```python
@dataclass
class CommentBlock:
    commenter_number: int
    comment_excerpt: str
    has_individual_response: bool
    raw_topic_clause: str        # "" when no IR
    individual_response_supplemental: str
def split_comment_blocks(text: str) -> list[CommentBlock]
```

**Approach:** Split on `Comment Excerpt from Commenter (\d+)`. Each block spans to the next such anchor or `Agency Discussion on` (topic boundary) or end. Within a block, find `Individual Response:\s*Please see Discussion[s]? on (?P<clause>.*?)\.` (DOTALL); excerpt = text before `Individual Response:` (strip running headers); supplemental = text after the clause's terminating period (strip running headers). No `Individual Response:` → `has_individual_response=False`, clause `""`, supplemental `""`.

- [ ] **Step 1: Failing test**
```python
# tests/test_rtc_responses.py
from stratification_scripts.rtc_parser import responses

TXT = """Comment Excerpt from Commenter 52
The public relies on EPA.
Individual Response: Please see Discussion on General Comments and Contaminant Groups. EPA agrees here.
Comment Excerpt from Commenter 99
An orphan excerpt with no response.
Comment Excerpt from Commenter 71
Another comment.
Individual Response: Please see Discussion on Length of CCL 5.
"""

def test_blocks_capture_excerpt_clause_supplemental():
    blocks = {b.commenter_number: b for b in responses.split_comment_blocks(TXT)}
    assert "public relies on EPA" in blocks[52].comment_excerpt
    assert blocks[52].raw_topic_clause.strip() == "General Comments and Contaminant Groups"
    assert "EPA agrees here" in blocks[52].individual_response_supplemental
    assert blocks[71].raw_topic_clause.strip() == "Length of CCL 5"

def test_orphan_excerpt_has_no_individual_response():
    blocks = {b.commenter_number: b for b in responses.split_comment_blocks(TXT)}
    assert blocks[99].has_individual_response is False
    assert blocks[99].raw_topic_clause == ""
    assert len(blocks) == 3  # off-by-one does not swallow the orphan
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement the anchored splitter + per-block regex. Anchor boundaries with `re.finditer(r"Comment Excerpt from Commenter (\d+)", text)`; slice each block to the next anchor; also cut a block at a `Agency Discussion on` line if present. Parse the IR clause with `re.search(r"Individual Response:\s*Please see Discussions? on (.*?)\.", block, re.DOTALL)`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(rtc): comment-block splitter with off-by-one tolerance`.

---

### Task 9: crosswalk.py (assemble records)

**Files:** Create `stratification_scripts/rtc_parser/crosswalk.py`; Test `tests/test_rtc_crosswalk.py`.

**Interfaces — Consumes:** `Commenter`, `CommentBlock`, `CommentRecord`, `topics.resolve_refs`, `split_topic_discussions`. **Produces:** `assemble(commenters: list[Commenter], blocks: list[CommentBlock], discussions: dict[str, str]) -> list[CommentRecord]`.

**Approach:** index commenters by number. For each block: `document_id`/name/org from the commenter index (None/"" if absent); `topic_refs = resolve_refs(raw_topic_clause)` when `has_individual_response` else `[]`; `topic_discussions = {ref.canonical: discussions[ref.canonical] for ref in topic_refs if ref.resolved and ref.canonical in discussions}`.

- [ ] **Step 1: Failing test**
```python
def test_assemble_joins_docid_topics_and_discussion():
    from stratification_scripts.rtc_parser.models import Commenter
    from stratification_scripts.rtc_parser.responses import CommentBlock
    commenters = [Commenter(52, "EPA-HQ-OW-2018-0594-0052", "A", "B", "Org")]
    blocks = [CommentBlock(52, "excerpt", True, "General Comments and PFAS", "supp"),
              CommentBlock(99, "orphan", False, "", "")]
    disc = {"General Comments": "GC text",
            "Per- and Polyfluoroalkyl substances (PFAS)": "PFAS text"}
    recs = {r.commenter_number: r for r in crosswalk.assemble(commenters, blocks, disc)}
    assert recs[52].document_id == "EPA-HQ-OW-2018-0594-0052"
    assert recs[52].topic_discussions["General Comments"] == "GC text"
    assert {t.raw for t in recs[52].topic_refs} == {"General Comments", "PFAS"}
    assert recs[99].document_id is None and recs[99].topic_refs == []
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `assemble` per the approach.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(rtc): assemble per-comment crosswalk records`.

---

### Task 10: cli.py + writers + __main__

**Files:** Create `stratification_scripts/rtc_parser/cli.py`; Test `tests/test_rtc_cli.py`.

**Interfaces — Produces:** `main(argv=None) -> int`; `parse_pdf(pdf_path, *, docket_prefix) -> tuple[list[Commenter], list[CommentRecord], dict]` (orchestrates extract→clean→parse); writer functions for the 4 outputs + manifest. CLI: `python -m stratification_scripts.rtc_parser parse --pdf <path> --slug ccl5 [--docket-prefix ...] [--out <dir>]`.

**Writers:** `commenters.csv` (polars), `crosswalk.jsonl` (one json per record; `topic_refs` as list of dicts), `crosswalk.csv` (flat: topics `"; "`-joined canonicals, `unresolved_topic_refs` `"; "`-joined raws), `topic_discussions.json`, `parse_manifest.json` (`{source_pdf_sha256, source_url, page_count, counts:{commenters, comments, topics, unresolved_refs}}`).

- [ ] **Step 1: Failing test** (drive orchestration on cleaned-text, and writers to tmp):
```python
def test_parse_and_write_outputs(tmp_path):
    # feed cleaned text directly through the parse helpers + writers
    from stratification_scripts.rtc_parser import cli
    out = cli.write_outputs(
        commenters=[...], records=[...], discussions={...},
        manifest={"source_pdf_sha256": "x", "source_url": "u", "page_count": 1,
                  "counts": {"commenters": 1, "comments": 1, "topics": 1, "unresolved_refs": 0}},
        out_dir=tmp_path)
    assert (tmp_path / "crosswalk.jsonl").exists()
    assert (tmp_path / "crosswalk.csv").exists()
    assert (tmp_path / "commenters.csv").exists()
    assert (tmp_path / "topic_discussions.json").exists()
    assert (tmp_path / "parse_manifest.json").exists()
```
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `parse_pdf` (extract_pages → strip_page_headers → parse_commenters / split_comment_blocks / split_topic_discussions → assemble), `write_outputs`, and `main` (argparse `parse` subcommand; refuse to overwrite a non-empty out dir like goldset's `write_seed_run`).
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(rtc): parse CLI + output writers + manifest`.

---

### Task 11: Real-slice fixture test + end-to-end acceptance

**Files:** Create `tests/fixtures/rtc_ccl5_slice.txt`; add tests to `tests/test_rtc_crosswalk.py` + `tests/test_rtc_cli.py`.

**Approach:** commit a small extracted-text slice of the real PDF spanning Exhibit 2 + one full topic (discussion + several comments incl. a multi-topic cross-ref). A committed test parses it and asserts stable real-data counts/joins. A separate e2e test runs `parse_pdf` on the full local PDF under `rtc/inputs/`, `@pytest.mark.skipif(not path.exists())`, asserting ~54 commenters / 114 comment records / 22 topics.

- [ ] **Step 1:** Generate the slice fixture from the real PDF (extract → strip_page_headers → hand-trim to the chosen pages); commit it.
- [ ] **Step 2: Real-slice test** asserting: Exhibit 2 rows parsed, ≥1 comment record with a resolved topic + attached discussion text, ≥0 unresolved surfaced.
- [ ] **Step 3: e2e skipif test** against the full local PDF.
- [ ] **Step 4:** Run `pytest tests/ -k rtc -v` → all green (e2e skips or passes if PDF present).
- [ ] **Step 5:** Run the CLI live against the real PDF; report counts; commit fixture+tests `test(rtc): real-slice fixture + skipif e2e acceptance`.

---

## Decision Ledger (living — canonical location for this build)

Adds to the same review banked for the goldset harness. Snapshot rationale also in the spec.

- **L1. Text state-machine over layout/color detection.** Extraction is clean+linear; robust literal anchors exist. Rejected fitz color/coordinate detection (brittle, couples to render). Reopen if a target RTC lacks textual anchors.
- **L2. `extract` (fitz/io) split from pure parse units.** Isolation + fixture-testability. Rejected monolithic parse-from-bytes. Reopen if parsing needs layout coords.
- **L3. Cross-ref fidelity: raw + canonical + resolved flag; never drop/force-match.** Prototype with no validation claims must flag uncertainty, not fabricate precision. Rejected verbatim-only (drops most) and greedy fuzzy (fake precision). Reopen when gold-set labels inform the alias map.
- **L4. Resolution order: exact → alias → unique-substring → unresolved.** Alias table holds only high-confidence expansions (PFAS, DBPs); unique-substring backstops; ambiguous/absent → flagged. Rejected multi-match substring (ambiguous) and edit-distance fuzzy (guessy). Reopen on labeled miss patterns.
- **L5. PDF gitignored + manifest committed.** Mirrors freeze; avoids LFS friction. Committed text fixtures drive deterministic tests; full e2e skipif-absent. Rejected committing the binary. Reopen if CI must run full e2e (commit an extracted-text snapshot then, not the binary).
- **L6. 114/113 off-by-one tolerated structurally.** Orphan excerpt → record with empty topics + `has_individual_response=False`, never positional zip. Reopen if the true cause is a false-positive anchor (fix the anchor, keep tolerance).
- **LA. Exhibit-2 first/last/org disambiguation heuristic (Task 5).** Person rows = first,last,org; anonymous/org-only rows = blank names + wrapped org. Heuristic chosen over positional-column assumptions because the linearized table drops blank name cells. Unit-tested on the three canonical shapes. Reopen if a real row shape breaks it (fix + add a fixture case).
- **L7. Docket prefix parameterized, default `EPA-HQ-OW-2018-0594`.** Avoids gratuitous hardcoding without pretending to generalize the layout. Reopen for a second RTC (new spec).

## Self-Review

- **Spec coverage:** Exhibit 2 (Task 5) ✓; Individual Response cross-refs (Tasks 7,8) ✓; Agency Topic Discussion disposition (Task 6) ✓; per-comment records Document ID→topics→discussion (Task 9) ✓; outputs+manifest (Task 10) ✓; storage/gitignore (Task 1) ✓; two-tier tests + e2e acceptance (Task 11) ✓; no-wiring/no-LLM/no-validation constraints honored throughout.
- **Placeholder scan:** Task 5 Step 3 and Task 10 Step 1 carry prose + `...` where the exact heuristic/fixtures are finalized during implementation against real shapes — acceptable because the *interfaces, tests, and decision rule* are pinned; the residual is tuning, not undefined behavior. All other steps carry runnable code.
- **Type consistency:** `Commenter`/`TopicRef`/`CommentRecord`/`CommentBlock` signatures consistent across Tasks 2,5,7,8,9; `resolve_refs`/`split_topic_discussions`/`split_comment_blocks`/`assemble` names stable across producer/consumer blocks.
