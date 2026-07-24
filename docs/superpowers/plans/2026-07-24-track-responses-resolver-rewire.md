# track_responses Resolver Rewire Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `track_responses`' faith-based `final_rule_document_number` lookup and its web-search fallback with the resolution layer — grounded judgment on resolver-found venues, typed envelope-relative absence for everything else, no open-web search anywhere.

**Architecture:** New pure module `stratification_scripts/makeup/resolution_routing.py` (CommentRef construction from pipeline rows, route derivation, typed row fields) with dependency injection so tests run on fakes; `track_responses_for_year` swaps its partition step to use it. The web-search fallback path and the old `build_grounded_cache` linkage are deleted at their call sites and definitions. Tier 2 (Gemini NPRM-vs-final comparison) is explicitly untouched.

**Tech Stack:** Python ≥3.10, polars, the `stratification_scripts.resolution` package (built 2026-07-24), pytest.

**Spec:** `docs/superpowers/specs/2026-07-24-comment-agent-and-fuel-architecture-design.md` §6 row 5, §8 Phase 2. Resolver contracts: `docs/superpowers/specs/2026-07-23-document-resolution-layer-design.md` (+ its Amendments section).

## Global Constraints

- **No open-web search anywhere in the measurement path.** After this plan, no code path in `track_responses.py` invokes a search-enabled LLM call for response tracking.
- **`UNKNOWN` never collapses into absence.** A row may say `response_found="no"` ONLY when its `resolution_status == "CONFIDENTLY_ABSENT"`.
- **Envelope semantics:** every absence row carries `envelope_version` (constant `"v1"` this plan) and `absence_reason`.
- **`response_source` values:** grounded rows keep `"fr_preamble"` (goldset frame compatibility until the separately-planned frame swap); absence rows use `"resolver_envelope"`; unknown rows use `"resolver_unknown"`. The value `"web_search"` is never written again.
- **Sampling/weights untouched.** `sample_comments_for_response_tracking`, `weight_map`, calibration, and Tier 2 are out of scope (Phases 3–4).
- **Do not modify the `resolution/` package** — consume its public API only.
- Commit messages are plain — no Co-Authored-By trailers.
- Baseline before Task 1: `pytest tests/ -q` → **203 passed** on `dev` at `f60d2c9`.

## Verified interfaces this plan consumes (from the built resolution package)

```python
from stratification_scripts.resolution import (
    AbsenceReason, CandidateDocument, Channel, CommentRef,
    ResolutionResult, ResponseEvidence, Status,
)
from stratification_scripts.resolution.resolver import DocumentResolver, qualifying_candidates
from stratification_scripts.resolution.cache import DocumentCache
# CommentRef(comment_id, comment_date, source_document, agency, rins: Tuple[str,...],
#            docket_id, packet_final_document)
# DocumentResolver(fr_client=..., reginfo_client=..., cache=...).resolve(ref) -> ResolutionResult
# resolver.cache -> DocumentCache; cache.extract(doc_number) -> Optional[ResponseExtract]
# qualifying_candidates(result) -> List[CandidateDocument]  (earliest publication first)
# ResolutionResult.status ∈ Status; .absence_reason ∈ Optional[AbsenceReason];
# .channels_run: Dict[Channel, str]; .candidates: List[CandidateDocument]
# CandidateDocument.document_number, .discovered_by: Channel, .response_evidence: ResponseEvidence
# An undated CommentRef (comment_date="") resolves to Status.UNKNOWN with all channels
# "skipped:no comment date" — the resolver's own guard; do not duplicate it.
```

## File Structure

```
stratification_scripts/makeup/resolution_routing.py   # NEW — pure routing core
stratification_scripts/makeup/track_responses.py      # MODIFY — swap partition, delete web path
tests/test_resolution_routing.py                      # NEW — Tasks 1-2
tests/test_track_responses_rewire.py                  # NEW — Tasks 3-4
tests/test_grounded_routing.py                        # DELETE in Task 5 (covers only the deleted build_grounded_cache)
```

---

### Task 1: `resolution_routing.py` — CommentRef from a pipeline row

**Files:**
- Create: `stratification_scripts/makeup/resolution_routing.py`
- Test: `tests/test_resolution_routing.py`

**Interfaces — Produces:**
- `ENVELOPE_VERSION = "v1"` (module constant)
- `ref_from_row(row: dict) -> CommentRef` — total function; missing dates yield `comment_date=""` (the resolver's undated guard handles it downstream)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_resolution_routing.py
from stratification_scripts.makeup.resolution_routing import ENVELOPE_VERSION, ref_from_row


def _row(**kw):
    base = dict(
        comment_id="DOT-OST-2024-0090-0049", document_number="2024-18496",
        agency="Transportation Department", posted_date="2024-09-23T14:00:00Z",
        receive_date="2024-09-22", rin="2105-AF05", rin_all="2105-AF05;2105-ZZ99",
        docket_id="Docket No. DOT-OST-2024-0090",
        final_rule_document_number="2025-02747",
    )
    base.update(kw)
    return base


def test_envelope_version_is_v1():
    assert ENVELOPE_VERSION == "v1"


def test_ref_from_full_row():
    ref = ref_from_row(_row())
    assert ref.comment_id == "DOT-OST-2024-0090-0049"
    assert ref.comment_date == "2024-09-23"          # posted_date wins, date part only
    assert ref.source_document == "2024-18496"
    assert ref.rins == ("2105-AF05", "2105-ZZ99")    # rin_all split, deduped, order kept
    assert ref.docket_id == "Docket No. DOT-OST-2024-0090"
    assert ref.packet_final_document == "2025-02747"


def test_receive_date_backfills_missing_posted():
    ref = ref_from_row(_row(posted_date=None))
    assert ref.comment_date == "2024-09-22"


def test_undated_row_yields_empty_comment_date():
    ref = ref_from_row(_row(posted_date=None, receive_date=""))
    assert ref.comment_date == ""                     # resolver guard handles it


def test_rin_fallback_when_no_rin_all():
    ref = ref_from_row(_row(rin_all=None))
    assert ref.rins == ("2105-AF05",)


def test_none_and_null_strings_are_cleaned():
    ref = ref_from_row(_row(final_rule_document_number="None", docket_id="null", rin="none", rin_all=None))
    assert ref.packet_final_document is None
    assert ref.docket_id is None
    assert ref.rins == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_resolution_routing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stratification_scripts.makeup.resolution_routing'`

- [ ] **Step 3: Write minimal implementation**

```python
# stratification_scripts/makeup/resolution_routing.py
"""Routing between the pipeline's comment rows and the resolution layer.

Pure functions only — no I/O, no clients. track_responses injects the resolver.
"""

from __future__ import annotations

from typing import Optional

from stratification_scripts.resolution import CommentRef

# The declared search envelope this routing implements. Bump ONLY when a new
# discovery channel ships (spec §6 row 5: envelope-relative absence).
ENVELOPE_VERSION = "v1"


def _clean(value) -> Optional[str]:
    text = str(value if value is not None else "").strip()
    return None if text.lower() in ("", "none", "null", "n/a") else text


def _rins(row: dict) -> tuple:
    raw = _clean(row.get("rin_all")) or _clean(row.get("rin")) or ""
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    return tuple(dict.fromkeys(p for p in parts if p and p.lower() not in ("none", "null")))


def ref_from_row(row: dict) -> CommentRef:
    """Build a CommentRef from one joined pipeline row (makeup ⋈ raw ⋈ FR)."""
    date = _clean(row.get("posted_date")) or _clean(row.get("receive_date")) or ""
    return CommentRef(
        comment_id=str(row.get("comment_id") or ""),
        comment_date=date[:10],
        source_document=str(row.get("document_number") or ""),
        agency=str(row.get("agency") or ""),
        rins=_rins(row),
        docket_id=_clean(row.get("docket_id")),
        packet_final_document=_clean(row.get("final_rule_document_number")),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_resolution_routing.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/makeup/resolution_routing.py tests/test_resolution_routing.py
git commit -m "feat(routing): CommentRef construction from pipeline rows + envelope constant"
```

---

### Task 2: Route derivation and typed row fields

**Files:**
- Modify: `stratification_scripts/makeup/resolution_routing.py`
- Test: `tests/test_resolution_routing.py` (append)

**Interfaces:**
- Consumes: `ResolutionResult`, `Status`, `qualifying_candidates`, `DocumentCache.extract` (Task 1's imports)
- Produces:
  - `@dataclass RoutedOutcome: kind: str ("grounded"|"absent"|"unknown"), candidate, extract, result`
  - `route_resolution(result, cache) -> RoutedOutcome`
  - `typed_fields(outcome) -> dict` — the new CSV columns, identical keys for every path:
    `resolution_status, absence_reason, envelope_version, resolved_document_number, discovered_by, resolution_channels`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_resolution_routing.py`:

```python
from stratification_scripts.resolution import (
    AbsenceReason, CandidateDocument, Channel, Relevance, ResponseEvidence,
    ResolutionResult, RuleClass, Status,
)
from stratification_scripts.makeup.resolution_routing import (
    RoutedOutcome, route_resolution, typed_fields,
)


def _cand(**kw):
    base = dict(
        document_number="2024-29990", publication_date="2024-12-18", type="Rule",
        action="Final rule.", title="t", agency_names=("Transportation Department",),
        rule_class=RuleClass.FINAL, rins=("2105-AF05",), docket_id=None,
        discovered_by=Channel.RIN_SEARCH, postdates_comment=True,
        relevance=Relevance.MATCH, response_evidence=ResponseEvidence.STRONG,
    )
    base.update(kw)
    return CandidateDocument(**base)


def _result(status, reason=None, candidates=(), channels=None):
    return ResolutionResult(
        comment_id="X-1", comment_date="2024-09-23", source_document="d",
        status=status, absence_reason=reason, candidates=list(candidates),
        agenda=None, channels_run=channels or {c: "ok" for c in Channel},
        resolved_at="2026-07-24T00:00:00",
    )


class FakeCache:
    def __init__(self, extracts):
        self._e = extracts

    def extract(self, document_number):
        return self._e.get(document_number)


class FakeExtract:
    grounded_text = "Comment: x Response: we agree"
    matched_header = "Comments and Responses"
    found_response_hd = True


def test_found_routes_to_grounded_with_first_qualifying_candidate():
    res = _result(Status.FOUND, candidates=[_cand()])
    out = route_resolution(res, FakeCache({"2024-29990": FakeExtract()}))
    assert out.kind == "grounded"
    assert out.candidate.document_number == "2024-29990"
    assert out.extract.grounded_text


def test_found_with_unreadable_extract_degrades_to_unknown_not_absent():
    res = _result(Status.FOUND, candidates=[_cand()])
    out = route_resolution(res, FakeCache({}))          # cache miss
    assert out.kind == "unknown"


def test_confidently_absent_routes_to_absent():
    res = _result(Status.CONFIDENTLY_ABSENT, reason=AbsenceReason.NO_VENUE_POSSIBLE,
                  candidates=[_cand(rule_class=RuleClass.DIRECT_FINAL, postdates_comment=False,
                                    response_evidence=ResponseEvidence.NONE)])
    out = route_resolution(res, FakeCache({}))
    assert out.kind == "absent"


def test_unknown_routes_to_unknown():
    res = _result(Status.UNKNOWN)
    assert route_resolution(res, FakeCache({})).kind == "unknown"


def test_typed_fields_grounded():
    res = _result(Status.FOUND, candidates=[_cand()])
    fields = typed_fields(route_resolution(res, FakeCache({"2024-29990": FakeExtract()})))
    assert fields["resolution_status"] == "FOUND"
    assert fields["absence_reason"] == ""
    assert fields["envelope_version"] == "v1"
    assert fields["resolved_document_number"] == "2024-29990"
    assert fields["discovered_by"] == "RIN_SEARCH"
    assert "PACKET_LINK:ok" in fields["resolution_channels"]


def test_typed_fields_absent_carries_reason():
    res = _result(Status.CONFIDENTLY_ABSENT, reason=AbsenceReason.RESPONSE_NOT_YET_PUBLISHED)
    fields = typed_fields(route_resolution(res, FakeCache({})))
    assert fields["resolution_status"] == "CONFIDENTLY_ABSENT"
    assert fields["absence_reason"] == "RESPONSE_NOT_YET_PUBLISHED"
    assert fields["resolved_document_number"] == ""


def test_unknown_never_renders_as_no():
    # The invariant, tested at the field level: unknown kind carries UNKNOWN status,
    # and (Task 4) only CONFIDENTLY_ABSENT rows may write response_found="no".
    res = _result(Status.UNKNOWN)
    fields = typed_fields(route_resolution(res, FakeCache({})))
    assert fields["resolution_status"] == "UNKNOWN"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_resolution_routing.py -v`
Expected: FAIL — `ImportError: cannot import name 'RoutedOutcome'`

- [ ] **Step 3: Write minimal implementation**

Append to `stratification_scripts/makeup/resolution_routing.py`:

```python
from dataclasses import dataclass

from stratification_scripts.resolution import ResolutionResult, Status
from stratification_scripts.resolution.resolver import qualifying_candidates


@dataclass
class RoutedOutcome:
    """Where one comment goes after resolution: grounded LLM read, typed absence, or unknown."""
    kind: str                      # "grounded" | "absent" | "unknown"
    result: ResolutionResult
    candidate=None                 # CandidateDocument for grounded, else None
    extract=None                   # ResponseExtract for grounded, else None


def route_resolution(result: ResolutionResult, cache) -> RoutedOutcome:
    """Map a ResolutionResult to its processing route.

    FOUND -> grounded on the first qualifying candidate whose extract is readable.
    A FOUND whose extract cannot be read degrades to UNKNOWN — never to absence.
    """
    if result.status is Status.FOUND:
        for candidate in qualifying_candidates(result):
            extract = cache.extract(candidate.document_number)
            if extract is not None and extract.grounded_text:
                return RoutedOutcome("grounded", result, candidate, extract)
        return RoutedOutcome("unknown", result)
    if result.status is Status.CONFIDENTLY_ABSENT:
        return RoutedOutcome("absent", result)
    return RoutedOutcome("unknown", result)


def typed_fields(outcome: RoutedOutcome) -> dict:
    """The typed CSV columns, schema-identical across all three routes."""
    r = outcome.result
    return {
        "resolution_status": r.status.value,
        "absence_reason": r.absence_reason.value if r.absence_reason else "",
        "envelope_version": ENVELOPE_VERSION,
        "resolved_document_number": outcome.candidate.document_number if outcome.candidate else "",
        "discovered_by": outcome.candidate.discovered_by.value if outcome.candidate else "",
        "resolution_channels": ";".join(f"{k.value}:{v}" for k, v in r.channels_run.items()),
    }
```

Note the dataclass fields `candidate`/`extract` need defaults without type annotations breaking dataclass ordering — implement exactly as:

```python
@dataclass
class RoutedOutcome:
    kind: str
    result: ResolutionResult
    candidate: object = None
    extract: object = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_resolution_routing.py -v`
Expected: 14 passed (6 from Task 1 + 8 new)

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/makeup/resolution_routing.py tests/test_resolution_routing.py
git commit -m "feat(routing): route derivation + typed envelope fields (FOUND/absent/unknown)"
```

---

### Task 3: Partition by resolution (the injectable seam)

**Files:**
- Modify: `stratification_scripts/makeup/resolution_routing.py`
- Test: `tests/test_resolution_routing.py` (append)

**Interfaces:**
- Consumes: `ref_from_row`, `route_resolution` (Tasks 1–2); a resolver object exposing `.resolve(ref)` and `.cache`
- Produces: `partition_by_resolution(comments: list[dict], resolver) -> tuple[list, list, list]`
  returning `(grounded, absent, unknown)` where each element is `(comment_row, RoutedOutcome)`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_resolution_routing.py`:

```python
from stratification_scripts.makeup.resolution_routing import partition_by_resolution


class FakeResolver:
    """Resolves by comment_id via a fixed table; counts calls for the dedup test."""
    def __init__(self, table, cache):
        self.table, self.cache, self.calls = table, cache, []

    def resolve(self, ref):
        self.calls.append(ref.comment_id)
        return self.table[ref.comment_id]


def test_partition_routes_each_comment_and_shares_the_cache():
    found = _result(Status.FOUND, candidates=[_cand()])
    absent = _result(Status.CONFIDENTLY_ABSENT, reason=AbsenceReason.NO_FINAL_RULE_PLANNED)
    unknown = _result(Status.UNKNOWN)
    resolver = FakeResolver(
        {"A": found, "B": absent, "C": unknown},
        FakeCache({"2024-29990": FakeExtract()}),
    )
    comments = [
        _row(comment_id="A"), _row(comment_id="B"), _row(comment_id="C"),
    ]
    grounded, absent_rows, unknown_rows = partition_by_resolution(comments, resolver)
    assert [c["comment_id"] for c, _ in grounded] == ["A"]
    assert [c["comment_id"] for c, _ in absent_rows] == ["B"]
    assert [c["comment_id"] for c, _ in unknown_rows] == ["C"]
    assert grounded[0][1].extract.grounded_text


def test_partition_resolves_each_comment_id_once():
    res = _result(Status.UNKNOWN)
    resolver = FakeResolver({"A": res}, FakeCache({}))
    partition_by_resolution([_row(comment_id="A"), _row(comment_id="A")], resolver)
    assert resolver.calls == ["A"]        # second row reuses the first resolution
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_resolution_routing.py -v`
Expected: FAIL — `ImportError: cannot import name 'partition_by_resolution'`

- [ ] **Step 3: Write minimal implementation**

Append to `stratification_scripts/makeup/resolution_routing.py`:

```python
def partition_by_resolution(comments, resolver):
    """Resolve every comment once and split by route.

    Returns (grounded, absent, unknown); each item is (comment_row, RoutedOutcome).
    The resolver's cross-row cache makes repeated documents cheap; repeated
    comment_ids (shouldn't happen, but joins have surprised us before — F22)
    are resolved once and reuse the outcome.
    """
    grounded, absent, unknown = [], [], []
    outcomes = {}
    for row in comments:
        cid = str(row.get("comment_id") or "")
        if cid not in outcomes:
            outcomes[cid] = route_resolution(resolver.resolve(ref_from_row(row)), resolver.cache)
        outcome = outcomes[cid]
        {"grounded": grounded, "absent": absent, "unknown": unknown}[outcome.kind].append((row, outcome))
    return grounded, absent, unknown
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_resolution_routing.py -v`
Expected: 16 passed

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/makeup/resolution_routing.py tests/test_resolution_routing.py
git commit -m "feat(routing): partition_by_resolution with per-comment memoization"
```

---

### Task 4: Wire into `track_responses_for_year`; typed rows on every save path

**Files:**
- Modify: `stratification_scripts/makeup/track_responses.py` (regions cited per step)
- Test: `tests/test_track_responses_rewire.py`

**Interfaces:**
- Consumes: `partition_by_resolution`, `typed_fields`, `ENVELOPE_VERSION` (Tasks 1–3); `DocumentResolver`, `DocumentCache` (resolution package); existing `save_responses_incremental`, `_save_grounded_results`
- Produces:
  - `_save_resolved_norun_rows(responses_csv, items, weight_map, df_comments, kind)` — writes absent/unknown rows WITHOUT any LLM call
  - `_save_grounded_results(...)` gains parameter `typed_by_id: dict[str, dict]` and merges those columns into each row
  - grounded rows carry `response_source="fr_preamble"`; absent rows `"resolver_envelope"` + `response_found="no"`; unknown rows `"resolver_unknown"` + `response_found="uncertain"`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_track_responses_rewire.py
import polars as pl

from stratification_scripts.makeup.resolution_routing import RoutedOutcome, typed_fields
from stratification_scripts.makeup.track_responses import _save_resolved_norun_rows
from stratification_scripts.resolution import AbsenceReason, Channel, ResolutionResult, Status


def _res(status, reason=None):
    return ResolutionResult(
        comment_id="B-1", comment_date="2024-09-23", source_document="d",
        status=status, absence_reason=reason, candidates=[], agenda=None,
        channels_run={c: "ok" for c in Channel}, resolved_at="t",
    )


def _comment(cid="B-1"):
    return {"comment_id": cid, "document_number": "2024-11111", "agency": "EPA",
            "lifecycle_stage": "NPRM_CLOSED", "rin": "2060-AV81", "attachment_text": ""}


def test_absent_rows_write_typed_no_without_llm(tmp_path):
    csv = tmp_path / "responses.csv"
    outcome = RoutedOutcome("absent", _res(Status.CONFIDENTLY_ABSENT, AbsenceReason.NO_FINAL_RULE_PLANNED))
    _save_resolved_norun_rows(csv, [(_comment(), outcome)], weight_map=None, df_comments=None, kind="absent")
    df = pl.read_csv(str(csv), infer_schema_length=None)
    row = df.to_dicts()[0]
    assert row["response_found"] == "no"
    assert row["response_source"] == "resolver_envelope"
    assert row["resolution_status"] == "CONFIDENTLY_ABSENT"
    assert row["absence_reason"] == "NO_FINAL_RULE_PLANNED"
    assert row["envelope_version"] == "v1"
    assert row["model"] == "none:resolver"


def test_unknown_rows_write_uncertain_never_no(tmp_path):
    csv = tmp_path / "responses.csv"
    outcome = RoutedOutcome("unknown", _res(Status.UNKNOWN))
    _save_resolved_norun_rows(csv, [(_comment("C-1"), outcome)], weight_map=None, df_comments=None, kind="unknown")
    row = pl.read_csv(str(csv), infer_schema_length=None).to_dicts()[0]
    assert row["response_found"] == "uncertain"
    assert row["response_source"] == "resolver_unknown"
    assert row["resolution_status"] == "UNKNOWN"


def test_appends_compatibly_onto_legacy_schema(tmp_path):
    # A pre-rewire CSV lacks the typed columns; save must merge, not crash.
    csv = tmp_path / "responses.csv"
    pl.DataFrame([{
        "comment_id": "OLD-1", "document_number": "d", "agency": "a",
        "response_found": "yes", "agency_decision": "accept", "response_text": "t",
        "response_location": "l", "reasoning": "r", "processed_at": "p", "model": "m",
        "comment_text_length": 0, "has_attachment": False, "lifecycle_stage": "s",
        "rin": "r", "response_sample_weight": 1.0, "response_source": "fr_preamble",
        "response_citation": "", "rtc_document_id": "",
    }]).write_csv(str(csv))
    outcome = RoutedOutcome("absent", _res(Status.CONFIDENTLY_ABSENT, AbsenceReason.NO_VENUE_POSSIBLE))
    _save_resolved_norun_rows(csv, [(_comment(), outcome)], weight_map=None, df_comments=None, kind="absent")
    df = pl.read_csv(str(csv), infer_schema_length=None)
    assert len(df) == 2
    assert set(["resolution_status", "absence_reason", "envelope_version"]).issubset(df.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_track_responses_rewire.py -v`
Expected: FAIL — `ImportError: cannot import name '_save_resolved_norun_rows'`

- [ ] **Step 3: Implement `_save_resolved_norun_rows` and extend `_save_grounded_results`**

In `stratification_scripts/makeup/track_responses.py`, add imports near the other project imports:

```python
from stratification_scripts.makeup.resolution_routing import (
    partition_by_resolution, ref_from_row, typed_fields,
)
```

Insert after `_save_grounded_results` (currently ends line ~677):

```python
def _save_resolved_norun_rows(
    responses_csv: Path,
    items: List,                       # [(comment_row, RoutedOutcome)]
    weight_map: Optional[Dict[str, float]],
    df_comments: Optional[pl.DataFrame],
    kind: str,                         # "absent" | "unknown"
) -> None:
    """Persist typed rows for comments that need NO LLM call.

    absent  -> response_found="no"        (envelope-relative: not found in any
               declared bin, searched cleanly — spec §6 row 5)
    unknown -> response_found="uncertain" (search incomplete; never collapsed to no)
    """
    verdict = "no" if kind == "absent" else "uncertain"
    source = "resolver_envelope" if kind == "absent" else "resolver_unknown"
    rows: List[Dict] = []
    for c, outcome in items:
        comment_id = str(c.get("comment_id"))
        fields = typed_fields(outcome)
        rows.append({
            "comment_id": comment_id,
            "document_number": str(c.get("document_number") or "N/A"),
            "agency": str(c.get("agency") or "N/A"),
            "response_found": verdict,
            "agency_decision": "uncertain" if kind == "unknown" else "no_response",
            "response_text": "N/A",
            "response_location": "N/A",
            "reasoning": (
                f"Resolver: {fields['resolution_status']}"
                + (f" ({fields['absence_reason']})" if fields["absence_reason"] else "")
                + f" under envelope {fields['envelope_version']}"
            ),
            "processed_at": datetime.now().isoformat(),
            "model": "none:resolver",
            "comment_text_length": 0,
            "has_attachment": bool(c.get("attachment_text")),
            "lifecycle_stage": str(c.get("lifecycle_stage") or "UNKNOWN"),
            "rin": str(c.get("rin") or "N/A"),
            "response_sample_weight": weight_map.get(comment_id, 1.0) if weight_map else 1.0,
            "response_source": source,
            "response_citation": "",
            "rtc_document_id": "",
            **fields,
        })
    save_responses_incremental(responses_csv, rows, df_comments)
```

Extend `_save_grounded_results` (line ~644): add parameter `typed_by_id: Optional[Dict[str, Dict]] = None` after `tracker_model`, and change the row construction's final lines from:

```python
            "response_source": "fr_preamble",
            "response_citation": (ext.matched_header if ext else "") or "",
            "rtc_document_id": "",
        })
```

to:

```python
            "response_source": "fr_preamble",
            "response_citation": (ext.matched_header if ext else "") or "",
            "rtc_document_id": "",
            **((typed_by_id or {}).get(comment_id, {})),
        })
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_track_responses_rewire.py -v`
Expected: 3 passed. Also run `.venv/bin/pytest tests/ -q` — expected: all green (the extended `_save_grounded_results` parameter is optional, so existing callers/tests still pass).

- [ ] **Step 5: Replace the partition inside `track_responses_for_year`**

In `track_responses_for_year`, replace the grounded-cache block and partition (currently lines ~865–925, from `# --- Primary-source grounding (Approach A): build per-rule grounded cache ---` through the `finally: if regs_client: regs_client.close()`) with:

```python
        # --- Resolution-layer routing (spec §6 row 5): resolver finds the venue;
        # grounded judgment reads it; everything else is typed, never web-searched. ---
        from stratification_scripts.federal_register.client import FederalRegisterClient
        from stratification_scripts.reginfo.client import RegInfoClient
        from stratification_scripts.resolution.resolver import DocumentResolver

        fr_client = FederalRegisterClient(max_retries=6, sleep_between=0.4)
        reginfo_client = RegInfoClient()
        resolver = DocumentResolver(fr_client=fr_client, reginfo_client=reginfo_client)
        try:
            grounded, absent_items, unknown_items = partition_by_resolution(
                comments_to_process, resolver
            )
            logger.info(
                f"Resolution routing: {len(grounded)} grounded, "
                f"{len(absent_items)} typed-absent, {len(unknown_items)} unknown "
                f"(envelope {ENVELOPE_VERSION}; no web search)"
            )

            grounded_items: List[tuple] = []
            grounded_meta: List[tuple] = []
            typed_by_id: Dict[str, Dict] = {}
            for c, outcome in grounded:
                full_text = extract_full_comment_text(c, client=regs_client, max_pages=config.max_comment_pages)
                meta = {
                    "comment_id": str(c.get("comment_id")),
                    "document_number": str(c.get("document_number", "N/A")),
                    "agency": str(c.get("agency", "N/A")),
                    "commenter_type": str(c.get("category", "N/A")),
                    "submission_date": str(c.get("posted_date", "N/A")),
                }
                grounded_items.append((full_text, outcome.extract.grounded_text, meta))
                grounded_meta.append((c, outcome.extract))
                typed_by_id[str(c.get("comment_id"))] = typed_fields(outcome)

            if grounded_items:
                logger.info(f"GROUNDED path: {len(grounded_items)} comments (resolver-found venues)")
                gres = asyncio.run(tracker.track_grounded_batch(grounded_items, max_concurrency=max_concurrency))
                _save_grounded_results(
                    responses_csv, gres, grounded_meta, weight_map,
                    df_raw if has_deduplication else None, tracker.model,
                    typed_by_id,
                )
            if absent_items:
                _save_resolved_norun_rows(
                    responses_csv, absent_items, weight_map,
                    df_raw if has_deduplication else None, "absent",
                )
            if unknown_items:
                _save_resolved_norun_rows(
                    responses_csv, unknown_items, weight_map,
                    df_raw if has_deduplication else None, "unknown",
                )
        finally:
            fr_client.close()
            reginfo_client.close()
            if regs_client:
                regs_client.close()
```

Also add `ENVELOPE_VERSION` to the routing import added in Step 3.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/pytest tests/ -q`
Expected: `tests/test_grounded_routing.py` may now fail if it exercises the replaced call path — everything else green. (Its deletion is Task 5; if it fails here, note it and proceed — do NOT fix it by restoring web-search code.)

- [ ] **Step 7: Commit**

```bash
git add stratification_scripts/makeup/track_responses.py tests/test_track_responses_rewire.py
git commit -m "feat(track): resolver-routed partition; typed absent/unknown rows; no web fallback"
```

---

### Task 5: Delete the dead web-search path and its tests; summary + smoke

**Files:**
- Modify: `stratification_scripts/makeup/track_responses.py`
- Delete: `tests/test_grounded_routing.py`
- Test: full suite + a `skipif`-gated live smoke

- [ ] **Step 1: Delete dead code**

In `track_responses.py`, delete entirely:
1. `build_grounded_cache` (lines ~604–641) — replaced by the resolver.
2. `process_responses_async` and its helper save block (the function starting line ~489 and the web-search row constructor around lines ~560–592) — the only caller was the fallback branch removed in Task 4. Before deleting, verify no other call sites: `grep -rn "process_responses_async\|build_grounded_cache" stratification_scripts/ tests/` — expected hits only in `track_responses.py` definitions and `tests/test_grounded_routing.py`.
3. `tests/test_grounded_routing.py` (covers only the two deleted functions).

Leave the provider clients' `RESPONSE_TRACKING_PROMPT` and `enable_search` flags untouched — clients are shared surface; only `track_responses` stops calling the search path. Leave Tier 2 (`_run_tier2_comparison`) untouched.

- [ ] **Step 2: Update the summary printer**

In `_print_response_summary`, after the `response_found` breakdown block (line ~987), insert:

```python
    if "resolution_status" in df_responses.columns:
        logger.info(f"\n{label} Resolution status breakdown:")
        for value in ["FOUND", "CONFIDENTLY_ABSENT", "UNKNOWN"]:
            count = df_responses.filter(pl.col("resolution_status") == value).shape[0]
            logger.info(f"  {value}: {count}")
        absents = df_responses.filter(pl.col("resolution_status") == "CONFIDENTLY_ABSENT")
        if len(absents) > 0:
            logger.info(f"{label} Absence reasons:")
            for value in ["NO_VENUE_POSSIBLE", "RESPONSE_NOT_YET_PUBLISHED", "NO_FINAL_RULE_PLANNED"]:
                count = absents.filter(pl.col("absence_reason") == value).shape[0]
                logger.info(f"  {value}: {count}")
```

- [ ] **Step 3: Add the no-web-search regression test**

Append to `tests/test_track_responses_rewire.py`:

```python
def test_no_web_search_symbols_remain():
    import stratification_scripts.makeup.track_responses as tr
    assert not hasattr(tr, "process_responses_async")
    assert not hasattr(tr, "build_grounded_cache")


def test_source_vocabulary_has_no_web_search(tmp_path):
    # The writable sources after the rewire; "web_search" must never be written again.
    import inspect
    import stratification_scripts.makeup.track_responses as tr
    source = inspect.getsource(tr)
    assert '"web_search"' not in source
```

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/pytest tests/ -q`
Expected: all green (count will be ~208–212 after deletions/additions; record the exact number in the commit message).

- [ ] **Step 5: Live smoke (manual, not CI)**

Run a one-comment live smoke against the 2024 data (any canonical comment id from the frozen snapshot works; NOAA's fixture id is a known-good FOUND):

```bash
.venv/bin/python -c "
from stratification_scripts.config import PipelineConfig
from stratification_scripts.makeup.track_responses import track_responses_for_year
track_responses_for_year(PipelineConfig(year=2024), limit=1)
" 2>&1 | tail -20
```

Expected: log shows `Resolution routing: ... (envelope v1; no web search)`; no exceptions; the responses CSV gains a row whose `resolution_status` is one of the three values. If API keys are absent in the environment, record that the smoke was skipped and why.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(track): delete web-search path + dead grounded cache; typed summary; regression tests"
```

---

## Not in this plan (deliberately)

- **Gold-set frame swap and redraw** (`status == FOUND` predicate) — next plan; depends on this one's columns existing.
- **RTC crosswalk as cascade step 0** — the parser exists; wiring it above the resolver is its own small plan.
- **Tier 2 removal or rework** — untouched here; flagged as an open question for Jonathan (it is not web search, but it rides the old linkage).
- **Sampling/weight fixes (F16–F18), estimator, universe weights** — Phases 3–4.
- **`AUDIT_FINDINGS` updates** — after Jonathan reviews the landed change.

## Self-Review

**Spec coverage:** §6 row 5 cascade → Tasks 3–4 (resolver → grounded → typed absence; crosswalk step explicitly deferred, listed above). Envelope semantics (versioned, inherited, `channels_run` per row) → Tasks 2, 4. UNKNOWN-never-collapses → route guard (Task 2 test), `_save_resolved_norun_rows` verdict mapping (Task 4 test), regression (Task 5). Web-search kill → Task 4 call-site removal + Task 5 deletion + source-scan regression. `response_source` compatibility for the goldset frame → Global Constraints + Task 4 (grounded rows keep `fr_preamble`).

**Placeholder scan:** none — every step carries code or an exact command. Two verify-before-delete greps are explicit commands with expected outputs.

**Type consistency:** `RoutedOutcome(kind, result, candidate, extract)` consistent across Tasks 2–4; `partition_by_resolution` returns `(comment_row, RoutedOutcome)` tuples consumed identically in Task 4; `typed_fields` keys match the CSV columns asserted in Task 4's tests; `_save_grounded_results(..., typed_by_id)` optional-parameter extension keeps the existing test suite valid until Task 5 deletes the obsolete file.
