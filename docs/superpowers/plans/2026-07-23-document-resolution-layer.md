# Document Resolution Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone resolver that answers *"where could a response to this comment live?"* — returning every candidate document it found, with provenance, rule classification, and a three-valued status — so the pipeline stops taking `final_doc_number` on faith and stops degrading silently into confident false negatives.

**Architecture:** New standalone package `stratification_scripts/resolution/`, mirroring `goldset`/`rtc_parser`/`freeze`. Pure decision functions (`classify`, `filters`, `evidence`, `status`) are separated from I/O (`channels`, `cache`, new search methods on `FederalRegisterClient`); `resolver.py` composes them behind dependency injection so the six-topology acceptance suite runs offline against recorded fixtures. Never imported by `stratification_scripts/cli.py`.

**Tech Stack:** Python ≥3.10, `requests` (via the existing `FederalRegisterClient`/`RegInfoClient`), `polars` for snapshot reads and tabular output, stdlib `re`/`dataclasses`/`enum`/`json`/`gzip`. pytest.

**Spec:** `docs/superpowers/specs/2026-07-23-document-resolution-layer-design.md`

## Global Constraints

- **No judgment.** The layer never decides whether an agency *responded*. It resolves venues only.
- **No pipeline wiring.** Never imported by `stratification_scripts/cli.py`. Invoked only via `python -m stratification_scripts.resolution`. Wiring into `track_responses` is a later, separately-measured change.
- **No Ruler B.** regulations.gov docket RTC PDFs are outside the declared envelope. Do not call `find_docket_rtc_documents()`.
- **Never collapse `UNKNOWN` into absence.** Any channel failure or skip, or an unreadable qualifying venue, is `UNKNOWN` — never `CONFIDENTLY_ABSENT`.
- **`response_evidence` is evidence, never a gate.** Qualification is `rule_class == FINAL ∧ postdates_comment ∧ relevance == MATCH`. The DOT row (`2024-29990`, `found_response_hd == False`, `method == suplinf_full`) is the named regression test for this.
- **Full-text search queries identifiers only, never subject terms.** Measured: `"Method 320"` → 83 unrelated rules; `"1808-IFC"` → 3, one exactly right.
- **The declared envelope is the five channels.** Every result records `channels_run` per row so the claim is reproducible and falsifiable.
- **Frozen snapshot `2026-07-15-ce44ac5` is read-only.** This module writes only under `resolution/`.
- **`*.csv` is Git-LFS-filtered in this repo** (`.gitattributes`). Test fixtures are `.json` / `.xml.gz` — never commit a `.csv` fixture.
- Commit messages end with the `Co-Authored-By: Claude Opus 4.8 (1M context)` trailer.

## Verified facts this plan is built on (checked live, 2026-07-23)

Do not re-derive these; they are pinned into the fixtures.

| Fact | Verification |
|---|---|
| `conditions[regulation_id_number]=1004-AF01` → 1 doc, `2024-27333`, `action="Direct final rule."` | live FR API |
| `conditions[docket_id]=CMS-1808-IFC` → 1 doc, `2024-22765`, `action="Interim final action with comment period."` | live FR API |
| `conditions[docket_id]=NOAA-NMFS-2023-0125` → `count: 0` (clean run, **not** a failure) | live FR API |
| `conditions[term]="1808-IFC"` → 3 docs incl. `2025-14681` | live FR API |
| `fields[]`: `document_number,title,type,action,publication_date,agencies,regulation_id_numbers,docket_ids,citation` all valid | live FR API |
| `agencies` is a list of dicts and **may contain `None` names** — `2024-29990` returns `['Transportation Department', None]` | live FR API |
| DOT packet link `2025-02747` is an **FCC** rule (`Radio Broadcasting Services`), RINs `[]` | live FR API |
| DOT true final rule `2024-29990`, `action="Final rule."`, pub `2024-12-18`, RIN `2105-AF05` | live FR API |
| reginfo RIN `2060-AV81` timetable HTML contains `<td headers='TimetableAction' …>Final Rule&nbsp;</td><td headers='TimetableDate' …>To Be Determined&nbsp;</td>` | live reginfo fetch |
| FR XML sizes: `2024-15931` 207 KB, `2024-29990` 81 KB, `2024-27333` 42 KB, **`2025-14681` 5.5 MB** (must be trimmed for fixtures) | live fetch |

---

## File Structure

```
stratification_scripts/reginfo/client.py     # MODIFY (Task 1 — prerequisite fix, own commit)
stratification_scripts/federal_register/client.py  # MODIFY (Task 7 — search methods)
stratification_scripts/config.py             # MODIFY (Task 2 — resolution path helpers)

stratification_scripts/resolution/
  __init__.py        # docstring + public exports
  __main__.py        # sys.exit(cli.main())
  models.py          # enums + CommentRef, CandidateDocument, AgendaStatus, ResolutionResult
  classify.py        # rule_class_from_action()                      [pure]
  filters.py         # postdates_comment(), relevance_of()           [pure]
  evidence.py        # response_evidence_from_extract()              [pure]
  status.py          # qualifies(), derive_status()                  [pure]
  cache.py           # DocumentCache — cross-row details + XML cache [I/O]
  channels.py        # the five discovery channels                   [I/O]
  resolver.py        # DocumentResolver — composition + fetch policy [I/O]
  inputs.py          # CommentRef construction from a frozen snapshot
  cli.py             # `resolve` command + jsonl/manifest writers

tests/
  test_reginfo_timetable.py            # Task 1
  test_resolution_models.py            # Task 2
  test_resolution_classify.py          # Task 3
  test_resolution_filters.py           # Task 4
  test_resolution_evidence.py          # Task 5
  test_resolution_status.py            # Task 6
  test_fr_client_search.py             # Task 7
  test_resolution_cache.py             # Task 8
  test_resolution_channels.py          # Tasks 9, 11
  test_resolution_resolver.py          # Tasks 10, 11
  test_resolution_acceptance.py        # Task 12 — the six topology rows
  test_resolution_cli.py               # Task 13
  fixtures/reginfo_2060-AV81_timetable.html   # Task 1 (real snippet)
  fixtures/resolution/
    _record.py                         # manual fixture recorder (not run by pytest)
    README.md                          # what was recorded, when, what was trimmed
    <comment_id>/input.json
    <comment_id>/expected.json
    <comment_id>/fr_doc_<docnum>.json
    <comment_id>/fr_rin_<rin>.json
    <comment_id>/fr_docket_<docket>.json
    <comment_id>/fr_term_<term>.json
    <comment_id>/agenda_<rin>.json
    <comment_id>/xml_<docnum>.xml.gz
```

---

### Task 1: Prerequisite — reginfo timetable must see non-date rows

Ships as its own commit ahead of the layer. Without it, `NO_FINAL_RULE_PLANNED` has no corroboration source and is dead code.

**Files:**
- Modify: `stratification_scripts/reginfo/client.py` (`_parse_html_response`, ~lines 409–502; `_parse_agenda_xml` entry construction ~line 565; new module-level helpers)
- Create: `tests/fixtures/reginfo_2060-AV81_timetable.html`
- Test: `tests/test_reginfo_timetable.py`

**Interfaces — Produces:**
- Every timetable entry dict gains a `date_raw: str` key (`""` when the date parsed to ISO). `date` keeps its existing meaning: ISO date or `""`. **No existing consumer changes** — `timetable_action_count`, `_safe_days_between`, and the chronological sort all key off `date`, which stays parseable-or-empty.
- `reginfo.client.has_undetermined_final_rule(agenda: Optional[dict]) -> bool`

**Why `date_raw` and not a raw-string `date`:** `extract_structured_timeline` counts `timetable_action_count` as entries with a non-empty `date`, and sorts on `date`. Stuffing `"To Be Determined"` into `date` would inflate the count and pollute the sort in every existing output column. The disconfirming signal is additive instead.

- [ ] **Step 1: Create the real-markup fixture**

Create `tests/fixtures/reginfo_2060-AV81_timetable.html` with this exact content (a verbatim slice of the live reginfo page for RIN 2060-AV81):

```html
<td>
    <b>Timetable:</b>
    <table class="generalTxt" width=95% CELLSPACING="1" bgcolor="darkgray" summary="This table contains timetable list.">
       <tr>
           <th id='TimetableAction' bgcolor="#CCCCCC"  scope="col" abbr="Action">Action</th>
           <th id='TimetableDate' bgcolor="#CCCCCC"  scope="col" abbr="Date">Date</th>
           <th id='FRC' bgcolor="#CCCCCC"  scope="col" abbr="FR">FR Cite</th>
       </tr>
               <tr>
                  <td headers='TimetableAction' bgcolor="#EFEFEF">NPRM&nbsp;</td>
                  <td headers='TimetableDate' bgcolor="#EFEFEF">03/01/2024&nbsp;</td>
                  <td headers='FRC' bgcolor="#EFEFEF">
                    <a class="pageSubNavTxt" href="javascript:leavePage('89 FR 15101', '', 'FR')">89 FR 15101</a>
                  </td>
               </tr>
               <tr>
                  <td headers='TimetableAction' bgcolor="#EFEFEF">Final Rule&nbsp;</td>
                  <td headers='TimetableDate' bgcolor="#EFEFEF">To Be Determined&nbsp;</td>
                  <td headers='FRC' bgcolor="#EFEFEF">
                  </td>
               </tr>
    </table>
    </td>
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_reginfo_timetable.py
from pathlib import Path

from stratification_scripts.reginfo.client import (
    RegInfoClient,
    has_undetermined_final_rule,
)

FIXTURE = Path(__file__).parent / "fixtures" / "reginfo_2060-AV81_timetable.html"

LONG_TERM_PAGE = (
    "<html><body>Long-Term Actions"
    + FIXTURE.read_text()
    + "</body></html>"
)


def _parse(html: str) -> dict:
    client = RegInfoClient.__new__(RegInfoClient)  # no network in __init__ path
    return client._parse_html_response(html, "2060-AV81")


def test_tbd_final_rule_row_is_captured():
    result = _parse(LONG_TERM_PAGE)
    actions = [e["action"] for e in result["timetable"]]
    assert "NPRM" in actions
    assert "FINAL RULE" in actions
    tbd = [e for e in result["timetable"] if e["action"] == "FINAL RULE"][0]
    assert tbd["date"] == ""                       # not parseable -> stays empty
    assert tbd["date_raw"] == "To Be Determined"   # ...but no longer invisible


def test_dated_row_still_parses_with_citation():
    result = _parse(LONG_TERM_PAGE)
    nprm = [e for e in result["timetable"] if e["action"] == "NPRM"][0]
    assert nprm["date"] == "2024-03-01"
    assert nprm["citation"] == "89 FR 15101"
    assert nprm["date_raw"] == ""


def test_every_entry_carries_date_raw():
    result = _parse(LONG_TERM_PAGE)
    assert all("date_raw" in e for e in result["timetable"])


def test_timetable_action_count_not_inflated_by_tbd_row():
    result = _parse(LONG_TERM_PAGE)
    timeline = RegInfoClient.extract_structured_timeline(result["timetable"])
    assert timeline["timetable_action_count"] == 1   # only the NPRM has a real date


def test_long_term_synthetic_entry_not_suppressed_by_a_dated_row():
    # The old gate was `not result["timetable"]`, so a page with an NPRM row
    # lost the undetermined signal entirely. It must survive now.
    result = _parse(LONG_TERM_PAGE)
    assert has_undetermined_final_rule(result) is True


def test_duplicate_tbd_rows_dedupe():
    doubled = LONG_TERM_PAGE.replace("</table>", """
               <tr>
                  <td headers='TimetableAction' bgcolor="#EFEFEF">Final Rule&nbsp;</td>
                  <td headers='TimetableDate' bgcolor="#EFEFEF">To Be Determined&nbsp;</td>
                  <td headers='FRC' bgcolor="#EFEFEF"></td>
               </tr>
    </table>""")
    result = _parse(doubled)
    finals = [e for e in result["timetable"] if e["action"] == "FINAL RULE"]
    assert len(finals) == 1


def test_fallback_parser_still_handles_pages_without_a_timetable_block():
    legacy = (
        "<html>Proposed Rule Stage"
        "<td>NPRM</td><td>03/01/2024</td><td>89 FR 15101</td>"
        "</html>"
    )
    result = _parse(legacy)
    assert [e["action"] for e in result["timetable"]] == ["NPRM"]
    assert result["timetable"][0]["date"] == "2024-03-01"
    assert result["timetable"][0]["date_raw"] == ""


def test_has_undetermined_final_rule_false_for_scheduled_final():
    agenda = {"timetable": [
        {"action": "NPRM", "date": "2024-03-01", "date_raw": "", "citation": ""},
        {"action": "FINAL RULE", "date": "2025-06-01", "date_raw": "", "citation": ""},
    ]}
    assert has_undetermined_final_rule(agenda) is False


def test_has_undetermined_final_rule_handles_none():
    assert has_undetermined_final_rule(None) is False
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_reginfo_timetable.py -v`
Expected: FAIL — `ImportError: cannot import name 'has_undetermined_final_rule'`.

- [ ] **Step 4: Implement the scoped row parser**

Add near the other module-level regexes in `stratification_scripts/reginfo/client.py` (above `class RegInfoClient`):

```python
# The timetable table region on a reginfo rule page.
_TIMETABLE_TABLE = re.compile(r"<b>\s*Timetable:\s*</b>(.*?)</table>", re.S | re.I)
_TIMETABLE_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S | re.I)
_TIMETABLE_CELL = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.S | re.I)
# Non-date date-cells that carry the disconfirming signal: no final rule scheduled.
_UNDETERMINED = re.compile(r"to\s+be\s+determined|next\s+action\s+undetermined|^\s*tbd\s*$", re.I)


def _cell_text(raw: str) -> str:
    """Strip tags/entities from one <td> and collapse whitespace."""
    txt = re.sub(r"<[^>]+>", " ", raw)
    txt = txt.replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", txt).strip()


def _parse_timetable_rows(html: str) -> Optional[List[dict]]:
    """Row-wise parse of the scoped Timetable table.

    Returns entries with action/date/date_raw/citation, or None when the page has
    no recognizable Timetable block (caller falls back to the legacy scan).

    Unlike the legacy two-cell regex, this keeps rows whose date cell is not a
    date ("To Be Determined") — the one field that can falsify "a final rule
    exists" — and reads the FR cite from the same row rather than "the next <td>".
    """
    block = _TIMETABLE_TABLE.search(html)
    if not block:
        return None
    entries: List[dict] = []
    for row in _TIMETABLE_ROW.finditer(block.group(1)):
        cells = [_cell_text(c) for c in _TIMETABLE_CELL.findall(row.group(1))]
        if len(cells) < 2:
            continue
        action, date_cell = cells[0], cells[1]
        cite_cell = cells[2] if len(cells) > 2 else ""
        if not action or action.lower() == "action":   # header row
            continue
        date_iso, date_raw = "", ""
        parts = date_cell.split("/")
        if len(parts) == 3 and parts[1] == "00":
            date_cell = f"{parts[0]}/01/{parts[2]}"
        try:
            date_iso = datetime.strptime(date_cell, "%m/%d/%Y").strftime("%Y-%m-%d")
        except ValueError:
            date_raw = date_cell
        fr_match = re.search(r"(\d+ FR \d+)", cite_cell)
        entries.append({
            "action": action.upper(),
            "date": date_iso,
            "date_raw": date_raw,
            "citation": fr_match.group(1) if fr_match else "",
        })
    return entries


def has_undetermined_final_rule(agenda: Optional[dict]) -> bool:
    """True when the agenda itself says no final rule is scheduled.

    This is the corroboration source for CONFIDENTLY_ABSENT/NO_FINAL_RULE_PLANNED.
    """
    if not agenda:
        return False
    for entry in agenda.get("timetable") or []:
        action = (entry.get("action") or "").upper()
        raw = entry.get("date_raw") or ""
        if action == "NEXT ACTION UNDETERMINED":
            return True
        if _UNDETERMINED.search(raw) and "FINAL" in action:
            return True
    return False
```

In `_parse_html_response`, immediately before the legacy `td_pattern` block (currently at line ~409), insert the scoped path and wrap the legacy loop as the fallback:

```python
        # Preferred: scoped, row-wise timetable parse (keeps non-date rows).
        scoped = _parse_timetable_rows(html)
        if scoped is not None:
            seen_keys = set()
            for entry in scoped:
                key = (entry["action"], entry["date"] or entry["date_raw"])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                result["timetable"].append(entry)
                if "WITHDRAWN" in entry["action"]:
                    result["withdrawn"] = True
```

Then guard the legacy loop so it only runs when the scoped parse found nothing:

```python
        if not result["timetable"]:
            # Legacy fallback for pages with no recognizable Timetable block.
            td_pattern = ...   # unchanged
            ...                # unchanged loop body, plus "date_raw": "" in the appended dict
```

Add `"date_raw": ""` to the entry dict at line ~474 (legacy loop), to both synthetic-entry appends (~lines 493 and 500), and to the XML entry at ~line 565.

Finally, un-gate the LONG_TERM synthetic entry (~line 490):

```python
        # A Long-Term rule whose undetermined signal did not survive parsing still
        # needs one. Previously gated on a fully-empty timetable, which silently
        # dropped the signal for any RIN that also had a real NPRM row.
        if result["stage"] == "LONG_TERM" and not has_undetermined_final_rule(result):
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_reginfo_timetable.py -v`
Expected: 9 passed.

- [ ] **Step 6: Run the full suite for regressions**

Run: `pytest tests/ -q`
Expected: all previously-passing tests still pass (baseline: 96 passed).

- [ ] **Step 7: Commit**

```bash
git add stratification_scripts/reginfo/client.py tests/test_reginfo_timetable.py tests/fixtures/reginfo_2060-AV81_timetable.html
git commit -m "fix(reginfo): capture non-date timetable rows (To Be Determined) + un-gate LONG_TERM signal"
```

---

### Task 2: Package scaffold, config paths, and the data contract

**Files:**
- Create: `stratification_scripts/resolution/__init__.py`, `__main__.py`, `models.py`
- Modify: `stratification_scripts/config.py` (add helpers next to `get_rtc_dir`, ~line 398)
- Modify: `.gitignore`
- Test: `tests/test_resolution_models.py`

**Interfaces — Produces:**
- `config.get_resolution_dir() -> Path` (`<project_root>/resolution`); `config.get_resolution_run_path(run_id: str) -> Path`
- All enums and dataclasses listed in the code below. Every later task consumes these names exactly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_resolution_models.py
from stratification_scripts import config
from stratification_scripts.resolution.models import (
    AbsenceReason, AgendaStatus, CandidateDocument, Channel, CommentRef,
    Relevance, ResolutionResult, ResponseEvidence, RuleClass, Status,
)


def test_resolution_paths_are_project_siblings():
    assert config.get_resolution_dir() == config.get_project_root() / "resolution"
    assert config.get_resolution_run_path("r1") == config.get_project_root() / "resolution" / "r1"


def test_enum_values_are_the_spec_strings():
    assert Channel.PACKET_LINK.value == "PACKET_LINK"
    assert Channel.FULLTEXT_SEARCH.value == "FULLTEXT_SEARCH"
    assert RuleClass.CONFIRMATION_OF_EFFECTIVE_DATE.value == "CONFIRMATION_OF_EFFECTIVE_DATE"
    assert Status.CONFIDENTLY_ABSENT.value == "CONFIDENTLY_ABSENT"
    assert AbsenceReason.NO_VENUE_POSSIBLE.value == "NO_VENUE_POSSIBLE"
    assert Relevance.AGENCY_MISMATCH.value == "AGENCY_MISMATCH"
    assert ResponseEvidence.WEAK.value == "WEAK"


def test_result_round_trips_to_json_dict():
    ref = CommentRef(
        comment_id="NOAA-NMFS-2023-0125-0016", comment_date="2024-03-22",
        source_document="2024-01120", agency="Commerce Department, National Oceanic",
        rins=("0648-BM40",), docket_id="NOAA-NMFS-2023-0125",
        packet_final_document="2024-15931",
    )
    cand = CandidateDocument(
        document_number="2024-15931", publication_date="2024-07-19", type="Rule",
        action="Final rule.", title="t", agency_names=("Commerce Department",),
        rule_class=RuleClass.FINAL, rins=("0648-BM40",), docket_id="NOAA-NMFS-2023-0125",
        discovered_by=Channel.PACKET_LINK, postdates_comment=True,
        relevance=Relevance.MATCH, response_evidence=ResponseEvidence.STRONG,
        response_header_matched=True, response_section_ref="Comments and Responses",
    )
    result = ResolutionResult(
        comment_id=ref.comment_id, comment_date=ref.comment_date,
        source_document=ref.source_document, status=Status.FOUND, absence_reason=None,
        candidates=[cand],
        agenda=AgendaStatus(rin="0648-BM40", stage="COMPLETED", timetable=[],
                            final_rule_undetermined=False, withdrawn=False,
                            fetched_at="2026-07-23T00:00:00", ok=True),
        channels_run={Channel.PACKET_LINK: "ok"},
        resolved_at="2026-07-23T00:00:00",
    )
    d = result.to_dict()
    assert d["status"] == "FOUND"
    assert d["absence_reason"] is None
    assert d["candidates"][0]["rule_class"] == "FINAL"
    assert d["channels_run"]["PACKET_LINK"] == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resolution_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stratification_scripts.resolution'`.

- [ ] **Step 3: Implement**

Add to `stratification_scripts/config.py`, directly after `get_rtc_inputs_dir`:

```python
def get_resolution_dir() -> Path:
    """Root for document-resolution run outputs (regenerable, gitignored)."""
    return get_project_root() / "resolution"


def get_resolution_run_path(run_id: str) -> Path:
    """Output directory for one resolution run."""
    return get_resolution_dir() / run_id
```

Create `stratification_scripts/resolution/__init__.py`:

```python
"""Document resolution layer.

Answers "where could a response to this comment live?" — returns every candidate
document found across five declared channels, with provenance, rule classification,
and a three-valued status. Never decides whether the agency actually responded.

Standalone: not imported by the pipeline cli.
"""

from .models import (  # noqa: F401
    AbsenceReason, AgendaStatus, CandidateDocument, Channel, CommentRef,
    Relevance, ResolutionResult, ResponseEvidence, RuleClass, Status,
)
```

Create `stratification_scripts/resolution/__main__.py`:

```python
import sys

from .cli import main

sys.exit(main())
```

Create `stratification_scripts/resolution/models.py`:

```python
"""Data contract for the document-resolution layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Channel(str, Enum):
    """The five discovery channels. Together they ARE the declared envelope."""
    PACKET_LINK = "PACKET_LINK"
    RIN_SEARCH = "RIN_SEARCH"
    DOCKET_SEARCH = "DOCKET_SEARCH"
    AGENDA = "AGENDA"
    FULLTEXT_SEARCH = "FULLTEXT_SEARCH"


class RuleClass(str, Enum):
    """Derived from the FR `action` field, never from `type`."""
    FINAL = "FINAL"
    DIRECT_FINAL = "DIRECT_FINAL"
    INTERIM_FINAL = "INTERIM_FINAL"
    CORRECTION = "CORRECTION"
    CONFIRMATION_OF_EFFECTIVE_DATE = "CONFIRMATION_OF_EFFECTIVE_DATE"
    PROPOSED = "PROPOSED"
    OTHER = "OTHER"


class Relevance(str, Enum):
    MATCH = "MATCH"
    AGENCY_MISMATCH = "AGENCY_MISMATCH"
    LINEAGE_MISMATCH = "LINEAGE_MISMATCH"


class ResponseEvidence(str, Enum):
    """Evidence that a candidate's preamble discusses comments. NOT a gate."""
    NONE = "NONE"
    WEAK = "WEAK"
    STRONG = "STRONG"


class Status(str, Enum):
    FOUND = "FOUND"
    CONFIDENTLY_ABSENT = "CONFIDENTLY_ABSENT"
    UNKNOWN = "UNKNOWN"


class AbsenceReason(str, Enum):
    NO_VENUE_POSSIBLE = "NO_VENUE_POSSIBLE"
    RESPONSE_NOT_YET_PUBLISHED = "RESPONSE_NOT_YET_PUBLISHED"
    NO_FINAL_RULE_PLANNED = "NO_FINAL_RULE_PLANNED"


@dataclass(frozen=True)
class CommentRef:
    """One comment, as the resolver needs to see it."""
    comment_id: str
    comment_date: str                       # ISO date, YYYY-MM-DD
    source_document: str                    # FR doc the comment was filed on
    agency: str                             # agency string from the FR row
    rins: Tuple[str, ...]
    docket_id: Optional[str]
    packet_final_document: Optional[str]    # upstream final_rule_document_number


@dataclass
class CandidateDocument:
    document_number: str
    publication_date: Optional[str]
    type: Optional[str]
    action: Optional[str]
    title: Optional[str]
    agency_names: Tuple[str, ...]
    rule_class: RuleClass
    rins: Tuple[str, ...]
    docket_id: Optional[str]
    discovered_by: Channel
    postdates_comment: bool
    relevance: Relevance
    response_evidence: ResponseEvidence = ResponseEvidence.NONE
    response_header_matched: Optional[bool] = None
    response_section_ref: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "document_number": self.document_number,
            "publication_date": self.publication_date,
            "type": self.type,
            "action": self.action,
            "title": self.title,
            "agency_names": list(self.agency_names),
            "rule_class": self.rule_class.value,
            "rins": list(self.rins),
            "docket_id": self.docket_id,
            "discovered_by": self.discovered_by.value,
            "postdates_comment": self.postdates_comment,
            "relevance": self.relevance.value,
            "response_evidence": self.response_evidence.value,
            "response_header_matched": self.response_header_matched,
            "response_section_ref": self.response_section_ref,
        }


@dataclass
class AgendaStatus:
    rin: Optional[str]
    stage: Optional[str]
    timetable: List[dict]
    final_rule_undetermined: bool
    withdrawn: bool
    fetched_at: str
    ok: bool          # False => AGENDA_NOT_FOUND; forces UNKNOWN on absence claims

    def to_dict(self) -> dict:
        return {
            "rin": self.rin, "stage": self.stage, "timetable": self.timetable,
            "final_rule_undetermined": self.final_rule_undetermined,
            "withdrawn": self.withdrawn, "fetched_at": self.fetched_at, "ok": self.ok,
        }


@dataclass
class ResolutionResult:
    comment_id: str
    comment_date: str
    source_document: str
    status: Status
    absence_reason: Optional[AbsenceReason]
    candidates: List[CandidateDocument] = field(default_factory=list)
    agenda: Optional[AgendaStatus] = None
    channels_run: Dict[Channel, str] = field(default_factory=dict)
    resolved_at: str = ""

    def to_dict(self) -> dict:
        return {
            "comment_id": self.comment_id,
            "comment_date": self.comment_date,
            "source_document": self.source_document,
            "status": self.status.value,
            "absence_reason": self.absence_reason.value if self.absence_reason else None,
            "candidates": [c.to_dict() for c in self.candidates],
            "agenda": self.agenda.to_dict() if self.agenda else None,
            "channels_run": {k.value: v for k, v in self.channels_run.items()},
            "resolved_at": self.resolved_at,
        }
```

Add to `.gitignore`:

```
# Document-resolution run outputs (regenerable; provenance in each run manifest)
/resolution/
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_resolution_models.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/resolution/ stratification_scripts/config.py .gitignore tests/test_resolution_models.py
git commit -m "feat(resolution): package scaffold + data contract + config paths"
```

---

### Task 3: Rule classification from the FR `action` field

**Files:**
- Create: `stratification_scripts/resolution/classify.py`
- Test: `tests/test_resolution_classify.py`

**Interfaces:**
- Consumes: `RuleClass` (Task 2)
- Produces: `rule_class_from_action(action: Optional[str], doc_type: Optional[str] = None) -> RuleClass`

**Why `action`, not `type`:** a plain final rule and a direct final rule both have `type == "Rule"`. Only `action` distinguishes them, and that distinction is the whole BLM fixture.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_resolution_classify.py
import pytest

from stratification_scripts.resolution.classify import rule_class_from_action
from stratification_scripts.resolution.models import RuleClass


@pytest.mark.parametrize("action,expected", [
    ("Final rule.", RuleClass.FINAL),
    ("Final rule", RuleClass.FINAL),
    ("Final rule; technical amendment.", RuleClass.FINAL),
    ("Direct final rule.", RuleClass.DIRECT_FINAL),
    ("Direct final rule; request for comments.", RuleClass.DIRECT_FINAL),
    ("Interim final rule.", RuleClass.INTERIM_FINAL),
    ("Interim final action with comment period.", RuleClass.INTERIM_FINAL),
    ("Final rule with request for comments.", RuleClass.INTERIM_FINAL),
    ("Final rule; confirmation of effective date.", RuleClass.CONFIRMATION_OF_EFFECTIVE_DATE),
    ("Confirmation of effective date.", RuleClass.CONFIRMATION_OF_EFFECTIVE_DATE),
    ("Final rule; correction", RuleClass.CORRECTION),
    ("Correcting amendment.", RuleClass.CORRECTION),
    ("Notice of proposed rulemaking (NPRM).", RuleClass.PROPOSED),
    ("Proposed rule.", RuleClass.PROPOSED),
    ("Supplemental notice of proposed rulemaking.", RuleClass.PROPOSED),
    ("Notification of enforcement discretion.", RuleClass.OTHER),
    ("Notice of availability of fishery management plan.", RuleClass.OTHER),
    ("", RuleClass.OTHER),
    (None, RuleClass.OTHER),
])
def test_rule_class_from_action(action, expected):
    assert rule_class_from_action(action) is expected


def test_doc_type_only_backfills_when_action_is_empty():
    assert rule_class_from_action(None, doc_type="Rule") is RuleClass.FINAL
    assert rule_class_from_action(None, doc_type="Proposed Rule") is RuleClass.PROPOSED
    # An explicit action always wins over doc_type.
    assert rule_class_from_action("Direct final rule.", doc_type="Rule") is RuleClass.DIRECT_FINAL


def test_deferred_response_variants_are_not_final():
    # These can carry responses to EARLIER-stage comments, so they must never
    # qualify as FINAL — but they do block a confident-absence claim (Task 6).
    for action in ["Interim final action with comment period.",
                   "Final rule with request for comments."]:
        assert rule_class_from_action(action) is RuleClass.INTERIM_FINAL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resolution_classify.py -v`
Expected: FAIL — `ModuleNotFoundError: ...resolution.classify`.

- [ ] **Step 3: Implement**

```python
# stratification_scripts/resolution/classify.py
"""Rule classification from the Federal Register `action` field.

`type` is useless here: a plain final rule and a direct final rule are both
type == "Rule". Only `action` separates a document that can carry a
comments-and-responses discussion from one that structurally cannot.
"""

from __future__ import annotations

import re
from typing import Optional

from .models import RuleClass

# Order matters: the first pattern that matches wins.
_PATTERNS = [
    (re.compile(r"confirmation\s+of\s+effective\s+date", re.I),
     RuleClass.CONFIRMATION_OF_EFFECTIVE_DATE),
    (re.compile(r"\bcorrect(ion|ing|ions)\b", re.I), RuleClass.CORRECTION),
    (re.compile(r"direct\s+final", re.I), RuleClass.DIRECT_FINAL),
    (re.compile(r"interim\s+final", re.I), RuleClass.INTERIM_FINAL),
    # A "final rule" that reopens comments is a deferred-response variant: the
    # answer to its own comments lands in a LATER document.
    (re.compile(r"final\s+rule.*(request\s+for\s+comment|comment\s+period)", re.I),
     RuleClass.INTERIM_FINAL),
    (re.compile(r"\b(proposed\s+rule|nprm|proposed\s+rulemaking)\b", re.I),
     RuleClass.PROPOSED),
    (re.compile(r"final\s+(rule|action)", re.I), RuleClass.FINAL),
]

_DOC_TYPE_FALLBACK = {
    "Rule": RuleClass.FINAL,
    "Proposed Rule": RuleClass.PROPOSED,
}


def rule_class_from_action(action: Optional[str], doc_type: Optional[str] = None) -> RuleClass:
    """Classify an FR document from its `action` string.

    doc_type is a fallback ONLY when action is missing; an explicit action always
    wins, because doc_type cannot distinguish a direct final rule from a final rule.
    """
    text = (action or "").strip()
    if not text:
        return _DOC_TYPE_FALLBACK.get((doc_type or "").strip(), RuleClass.OTHER)
    for pattern, rule_class in _PATTERNS:
        if pattern.search(text):
            return rule_class
    return RuleClass.OTHER
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_resolution_classify.py -v`
Expected: 23 passed.

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/resolution/classify.py tests/test_resolution_classify.py
git commit -m "feat(resolution): rule classification from the FR action field"
```

---

### Task 4: Chronology and relevance filters

**Files:**
- Create: `stratification_scripts/resolution/filters.py`
- Test: `tests/test_resolution_filters.py`

**Interfaces:**
- Consumes: `Channel`, `CommentRef`, `Relevance` (Task 2)
- Produces:
  - `postdates_comment(publication_date: Optional[str], comment_date: str) -> bool`
  - `normalize_agency(name: Optional[str]) -> str`
  - `agency_matches(candidate_agency_names: Sequence[Optional[str]], ref_agency: str) -> bool`
  - `relevance_of(*, discovered_by: Channel, agency_names: Sequence[Optional[str]], rins: Sequence[str], docket_id: Optional[str], ref: CommentRef) -> Relevance`

**Lineage scoping (spec, "Relevance check (channel 1)"):** channels 2/3/5 query *by* an identifier drawn from the comment's own lineage, so a hit is itself lineage evidence — the CMS response lives under a different RIN and a different docket and must not be thrown away. The lineage half of the check therefore applies to `PACKET_LINK` only. The agency half applies to every channel.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_resolution_filters.py
from stratification_scripts.resolution.filters import (
    agency_matches, normalize_agency, postdates_comment, relevance_of,
)
from stratification_scripts.resolution.models import Channel, CommentRef, Relevance

DOT_REF = CommentRef(
    comment_id="DOT-OST-2024-0090-0049", comment_date="2024-09-23",
    source_document="2024-18496", agency="Transportation Department",
    rins=("2105-AF05",), docket_id="Docket No. DOT-OST-2024-0090",
    packet_final_document="2025-02747",
)

CMS_REF = CommentRef(
    comment_id="CMS-2024-0131-6043", comment_date="2024-12-03",
    source_document="2024-22765", agency="Health and Human Services Department",
    rins=("0938-AV34",), docket_id="CMS-1808-IFC", packet_final_document=None,
)


def test_postdates_comment():
    assert postdates_comment("2024-12-18", "2024-09-23") is True
    assert postdates_comment("2024-09-23", "2024-09-23") is True   # same day counts
    assert postdates_comment("2024-08-22", "2024-09-23") is False
    assert postdates_comment(None, "2024-09-23") is False
    assert postdates_comment("garbage", "2024-09-23") is False


def test_normalize_agency_strips_case_and_punctuation():
    assert normalize_agency("Health and Human Services Department") == "health and human services department"
    assert normalize_agency("  Transportation Department ") == "transportation department"
    assert normalize_agency(None) == ""


def test_agency_matches_tolerates_none_entries():
    # The live FR API returns ['Transportation Department', None] for 2024-29990.
    assert agency_matches(["Transportation Department", None], "Transportation Department") is True


def test_agency_matches_on_multi_agency_ref_string():
    # The frozen FR rows carry comma-joined agency strings.
    assert agency_matches(
        ["Commerce Department", "National Oceanic and Atmospheric Administration"],
        "Commerce Department, National Oceanic and Atmospheric Administration",
    ) is True


def test_agency_matches_parent_child():
    # CMS documents list both HHS (parent) and CMS (child); the ref names one.
    assert agency_matches(
        ["Health and Human Services Department", "Centers for Medicare & Medicaid Services"],
        "Health and Human Services Department",
    ) is True


def test_packet_link_to_wrong_agency_is_rejected():
    # DOT's packet link points at an FCC radio-broadcasting rule.
    assert relevance_of(
        discovered_by=Channel.PACKET_LINK,
        agency_names=["Federal Communications Commission"],
        rins=[], docket_id="DA 25-120", ref=DOT_REF,
    ) is Relevance.AGENCY_MISMATCH


def test_packet_link_same_agency_wrong_lineage_is_rejected():
    assert relevance_of(
        discovered_by=Channel.PACKET_LINK,
        agency_names=["Transportation Department"],
        rins=["2105-ZZ99"], docket_id="Docket No. DOT-OST-2099-1111", ref=DOT_REF,
    ) is Relevance.LINEAGE_MISMATCH


def test_packet_link_matching_rin_passes():
    assert relevance_of(
        discovered_by=Channel.PACKET_LINK,
        agency_names=["Transportation Department"],
        rins=["2105-AF05"], docket_id=None, ref=DOT_REF,
    ) is Relevance.MATCH


def test_fulltext_hit_under_a_different_rin_and_docket_still_matches():
    # The CMS response (2025-14681) lives under a different RIN and docket.
    # Channel provenance IS the lineage evidence — rejecting it would delete the
    # single topology this layer exists to recover.
    assert relevance_of(
        discovered_by=Channel.FULLTEXT_SEARCH,
        agency_names=["Health and Human Services Department",
                      "Centers for Medicare & Medicaid Services"],
        rins=["0938-AV53"], docket_id="CMS-1809-F", ref=CMS_REF,
    ) is Relevance.MATCH


def test_fulltext_hit_from_another_agency_is_still_rejected():
    assert relevance_of(
        discovered_by=Channel.FULLTEXT_SEARCH,
        agency_names=["Federal Communications Commission"],
        rins=[], docket_id=None, ref=CMS_REF,
    ) is Relevance.AGENCY_MISMATCH


def test_unknown_candidate_agency_is_not_treated_as_a_mismatch():
    # Missing metadata must not manufacture a rejection.
    assert relevance_of(
        discovered_by=Channel.RIN_SEARCH,
        agency_names=[], rins=["2105-AF05"], docket_id=None, ref=DOT_REF,
    ) is Relevance.MATCH
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resolution_filters.py -v`
Expected: FAIL — `ModuleNotFoundError: ...resolution.filters`.

- [ ] **Step 3: Implement**

```python
# stratification_scripts/resolution/filters.py
"""Chronology and relevance filters.

Chronology: a candidate can only be a response if it postdates the comment.
Relevance: a link is only usable if it belongs to this rulemaking's lineage —
the check that would have caught DOT's packet link pointing at an FCC rule.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Optional, Sequence

from ..federal_register.client import normalize_docket_id
from .models import Channel, CommentRef, Relevance

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_SPACE = re.compile(r"\s+")


def _iso(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def postdates_comment(publication_date: Optional[str], comment_date: str) -> bool:
    """True when the candidate was published on or after the comment date.

    Unparseable or missing dates are False: an unknown date is not evidence of
    a valid chronology. Observed failure this prevents — the CMS RIN's three
    Rule-type documents all predate the comment.
    """
    pub, com = _iso(publication_date), _iso(comment_date)
    if pub is None or com is None:
        return False
    return pub >= com


def normalize_agency(name: Optional[str]) -> str:
    if not name:
        return ""
    return _SPACE.sub(" ", _PUNCT.sub(" ", str(name).lower())).strip()


def agency_matches(candidate_agency_names: Sequence[Optional[str]], ref_agency: str) -> bool:
    """True when any candidate agency overlaps the comment's agency string.

    The frozen FR rows carry comma-joined agency strings ("Commerce Department,
    National Oceanic ...") while the API returns a list that may contain None
    entries. Overlap either way counts — parent/child pairs (HHS/CMS) match.
    An EMPTY candidate list is not a mismatch; the caller treats it as unknown.
    """
    ref_parts = [normalize_agency(p) for p in str(ref_agency or "").split(",")]
    ref_parts = [p for p in ref_parts if p]
    cand_parts = [normalize_agency(n) for n in candidate_agency_names]
    cand_parts = [p for p in cand_parts if p]
    if not ref_parts or not cand_parts:
        return True
    for c in cand_parts:
        for r in ref_parts:
            if c == r or c in r or r in c:
                return True
    return False


def relevance_of(
    *,
    discovered_by: Channel,
    agency_names: Sequence[Optional[str]],
    rins: Sequence[str],
    docket_id: Optional[str],
    ref: CommentRef,
) -> Relevance:
    """Classify a candidate's relevance to the comment's rulemaking.

    Agency check: every channel. Lineage check: PACKET_LINK only — channels 2/3/5
    query BY an identifier taken from the comment's own lineage, so a hit is
    itself lineage evidence (the CMS response sits under a different RIN and
    docket and must survive).
    """
    if not agency_matches(agency_names, ref.agency):
        return Relevance.AGENCY_MISMATCH
    if discovered_by is not Channel.PACKET_LINK:
        return Relevance.MATCH
    ref_rins = {r.strip().upper() for r in ref.rins if r}
    cand_rins = {str(r).strip().upper() for r in rins if r}
    if ref_rins and cand_rins and (ref_rins & cand_rins):
        return Relevance.MATCH
    ref_docket = normalize_docket_id(ref.docket_id)
    cand_docket = normalize_docket_id(docket_id)
    if ref_docket and cand_docket and ref_docket == cand_docket:
        return Relevance.MATCH
    if not cand_rins and not cand_docket:
        # No lineage metadata at all — unknowable, not disqualifying.
        return Relevance.MATCH
    return Relevance.LINEAGE_MISMATCH
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_resolution_filters.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/resolution/filters.py tests/test_resolution_filters.py
git commit -m "feat(resolution): chronology + relevance filters (channel-scoped lineage)"
```

---

### Task 5: Response evidence — graded, never a gate

**Files:**
- Create: `stratification_scripts/resolution/evidence.py`
- Test: `tests/test_resolution_evidence.py`

**Interfaces:**
- Consumes: `ResponseEvidence` (Task 2); `ResponseExtract` and `DENSITY_KW` from `stratification_scripts.makeup.fr_response_extractor`
- Produces: `response_evidence_from_extract(extract: Optional[ResponseExtract]) -> ResponseEvidence`; `density_per_1k(text: str) -> float`; module constant `STRONG_DENSITY_PER_1K = 2.0`

**The defect this replaces:** `extract_response_section()` sets `found_response_hd=True` only when its `RESP_HD` header regex matches. Its `suplinf_full` and `comment_density` fallbacks return *real* response text with `found_response_hd=False`. DOT `2024-29990` is a hand-confirmed genuine response with `found_response_hd == False` and 68k chars of response text. Gating on the header flag would have manufactured a false `CONFIDENTLY_ABSENT` — the exact silent failure this layer exists to kill.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_resolution_evidence.py
from stratification_scripts.makeup.fr_response_extractor import ResponseExtract
from stratification_scripts.resolution.evidence import (
    density_per_1k, response_evidence_from_extract,
)
from stratification_scripts.resolution.models import ResponseEvidence

DENSE = (
    "Comment: Several commenters argued the rule is too costly. "
    "Response: We disagree with the commenters and are adopting the provision. "
    "In response to comment, the agency considered the alternative. "
) * 20

SPARSE = ("This rule adopts technical amendments to the table of contents. " * 60)


def test_header_match_is_strong():
    ext = ResponseExtract("short text", "response_hd", "Comments and Responses", True, 5000, 10)
    assert response_evidence_from_extract(ext) is ResponseEvidence.STRONG


def test_dot_case_no_header_but_dense_text_is_strong():
    # DOT 2024-29990: method=suplinf_full, found_response_hd=False, real responses.
    ext = ResponseExtract(DENSE, "suplinf_full", None, False, len(DENSE), len(DENSE))
    assert response_evidence_from_extract(ext) is ResponseEvidence.STRONG


def test_low_density_text_is_weak():
    ext = ResponseExtract(SPARSE, "suplinf_full", None, False, len(SPARSE), len(SPARSE))
    assert response_evidence_from_extract(ext) is ResponseEvidence.WEAK


def test_empty_extract_is_none():
    ext = ResponseExtract("", "no_preamble", None, False, 0, 0)
    assert response_evidence_from_extract(ext) is ResponseEvidence.NONE


def test_missing_extract_is_none():
    assert response_evidence_from_extract(None) is ResponseEvidence.NONE


def test_density_per_1k_is_a_rate_not_a_count():
    assert density_per_1k(DENSE) > density_per_1k(SPARSE)
    assert density_per_1k("") == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resolution_evidence.py -v`
Expected: FAIL — `ModuleNotFoundError: ...resolution.evidence`.

- [ ] **Step 3: Implement**

```python
# stratification_scripts/resolution/evidence.py
"""Graded response evidence, derived from the extract — never from the header flag.

extract_response_section() sets found_response_hd=True only when its RESP_HD
header regex matches; the suplinf_full and comment_density fallbacks return real
response text with the flag False (DOT 2024-29990: 68k chars of genuine agency
responses, flag False). The flag is also wrong in the other direction — CMS
2025-14681 matched a section other than the one that mattered. So the flag is one
input among several, and evidence NEVER gates qualification.
"""

from __future__ import annotations

from typing import Optional

from ..makeup.fr_response_extractor import DENSITY_KW, ResponseExtract
from .models import ResponseEvidence

# Comment/response keyword hits per 1,000 characters. Calibrated so a preamble
# that actually walks through comments clears it while a technical-amendment
# preamble that merely mentions the comment period does not.
STRONG_DENSITY_PER_1K = 2.0


def density_per_1k(text: str) -> float:
    """Comment-discussion keyword hits per 1,000 characters of grounded text."""
    if not text:
        return 0.0
    return len(DENSITY_KW.findall(text)) / (len(text) / 1000.0)


def response_evidence_from_extract(extract: Optional[ResponseExtract]) -> ResponseEvidence:
    """Grade the evidence that a candidate's preamble discusses comments.

    STRONG - a response header matched, OR the grounded text is dense in
             comment/response language.
    WEAK   - grounded text exists but is sparse.
    NONE   - no preamble, or empty grounded text.
    """
    if extract is None or not extract.grounded_text:
        return ResponseEvidence.NONE
    if extract.found_response_hd:
        return ResponseEvidence.STRONG
    if density_per_1k(extract.grounded_text) >= STRONG_DENSITY_PER_1K:
        return ResponseEvidence.STRONG
    return ResponseEvidence.WEAK
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_resolution_evidence.py -v`
Expected: 6 passed.

If `test_low_density_text_is_weak` fails, do **not** loosen the DOT test — recheck `STRONG_DENSITY_PER_1K` against both fixture strings by printing `density_per_1k(DENSE)` and `density_per_1k(SPARSE)` and pick a threshold strictly between them, then record the measured numbers in the decision ledger.

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/resolution/evidence.py tests/test_resolution_evidence.py
git commit -m "feat(resolution): graded response evidence (header flag is not a gate)"
```

---

### Task 6: Status derivation and per-reason corroboration

The heart of the layer. Every review finding (F1–F4) lands here.

**Files:**
- Create: `stratification_scripts/resolution/status.py`
- Test: `tests/test_resolution_status.py`

**Interfaces:**
- Consumes: `AgendaStatus`, `CandidateDocument`, `Channel`, `Status`, `AbsenceReason`, `Relevance`, `ResponseEvidence`, `RuleClass` (Task 2)
- Produces:
  - `qualifies(candidate: CandidateDocument) -> bool`
  - `all_channels_clean(channels_run: Mapping[Channel, str]) -> bool`
  - `derive_status(candidates, agenda, channels_run) -> tuple[Status, Optional[AbsenceReason]]`
  - `CHANNEL_OK = "ok"` (the only value of `channels_run` that counts as clean; a zero-result channel is `"ok"`, a skip is `"skipped:<why>"`, a failure is `"failed:<why>"`)

**Rules, in evaluation order:**

1. `FOUND` — ≥1 qualifying candidate with `response_evidence != NONE`. Needs only its own evidence; an early exit after a hit is legal (F4).
2. `UNKNOWN` — a qualifying candidate exists but every one has `response_evidence == NONE` (venue found, unreadable).
3. `UNKNOWN` — a *non-FINAL* candidate postdates the comment with `response_evidence == STRONG` (F3: an interim final rule or "final rule with request for comments" can answer earlier-stage comments in its own preamble). **Keyed off grounded-text evidence, not `response_header_matched`** — otherwise F3's fix inherits F2's defect.
4. `UNKNOWN` — any channel not `"ok"`, or agenda missing/failed (`AGENDA_NOT_FOUND`).
5. `CONFIDENTLY_ABSENT` + `absence_reason`, if a reason corroborates; otherwise `UNKNOWN`.

**Precedence among absence reasons:** `NO_VENUE_POSSIBLE` → `NO_FINAL_RULE_PLANNED` → `RESPONSE_NOT_YET_PUBLISHED`. EPA Method 320 has both an NPRM and a TBD agenda; the fixture expects `NO_FINAL_RULE_PLANNED`, so the agenda reason outranks the not-yet reason. FBI has an NPRM with a normally-scheduled agenda, so it falls through to `RESPONSE_NOT_YET_PUBLISHED`.

**Corroboration table (per reason — a single global agenda clause is wrong; for `NO_VENUE_POSSIBLE` the agenda shows a *completed* action, so a global TBD requirement could never let the BLM fixture reach its expected status):**

| reason | corroboration |
|---|---|
| `NO_VENUE_POSSIBLE` | a candidate with `rule_class ∈ {DIRECT_FINAL, CONFIRMATION_OF_EFFECTIVE_DATE}`, `relevance == MATCH`, agenda not withdrawn, and that candidate's own `response_evidence != STRONG` |
| `NO_FINAL_RULE_PLANNED` | `agenda.final_rule_undetermined` or `agenda.stage == "LONG_TERM"` |
| `RESPONSE_NOT_YET_PUBLISHED` | a candidate with `rule_class ∈ {PROPOSED, INTERIM_FINAL}` and `relevance == MATCH` |

- [ ] **Step 1: Write the failing test**

```python
# tests/test_resolution_status.py
from stratification_scripts.resolution.models import (
    AbsenceReason, AgendaStatus, CandidateDocument, Channel, Relevance,
    ResponseEvidence, RuleClass, Status,
)
from stratification_scripts.resolution.status import (
    CHANNEL_OK, all_channels_clean, derive_status, qualifies,
)

ALL_CLEAN = {c: CHANNEL_OK for c in Channel}


def cand(**kw) -> CandidateDocument:
    base = dict(
        document_number="2024-00001", publication_date="2024-12-01", type="Rule",
        action="Final rule.", title="t", agency_names=("Agency",),
        rule_class=RuleClass.FINAL, rins=(), docket_id=None,
        discovered_by=Channel.RIN_SEARCH, postdates_comment=True,
        relevance=Relevance.MATCH, response_evidence=ResponseEvidence.STRONG,
    )
    base.update(kw)
    return CandidateDocument(**base)


def agenda(**kw) -> AgendaStatus:
    base = dict(rin="1234-AB56", stage="FINAL", timetable=[],
                final_rule_undetermined=False, withdrawn=False,
                fetched_at="2026-07-23T00:00:00", ok=True)
    base.update(kw)
    return AgendaStatus(**base)


def test_qualification_ignores_response_evidence():
    assert qualifies(cand(response_evidence=ResponseEvidence.NONE)) is True
    assert qualifies(cand(rule_class=RuleClass.DIRECT_FINAL)) is False
    assert qualifies(cand(postdates_comment=False)) is False
    assert qualifies(cand(relevance=Relevance.AGENCY_MISMATCH)) is False


def test_found_needs_only_its_own_evidence():
    # Two channels never ran; FOUND is still correct (F4).
    status, reason = derive_status([cand()], agenda(), {Channel.PACKET_LINK: CHANNEL_OK})
    assert status is Status.FOUND and reason is None


def test_dot_regression_weak_header_still_found():
    # DOT 2024-29990: no matching response header, but real response text.
    c = cand(response_evidence=ResponseEvidence.STRONG, response_header_matched=False)
    status, _ = derive_status([c], agenda(), ALL_CLEAN)
    assert status is Status.FOUND


def test_weak_evidence_on_a_qualifying_candidate_is_found_not_absent():
    status, _ = derive_status([cand(response_evidence=ResponseEvidence.WEAK)], agenda(), ALL_CLEAN)
    assert status is Status.FOUND


def test_qualifying_but_unreadable_is_unknown_never_absent():
    status, reason = derive_status(
        [cand(response_evidence=ResponseEvidence.NONE)], agenda(), ALL_CLEAN)
    assert status is Status.UNKNOWN and reason is None


def test_deferred_variant_with_real_text_blocks_absence():
    # An IFC can answer earlier-stage comments in its own preamble (F3).
    ifc = cand(rule_class=RuleClass.INTERIM_FINAL,
               action="Interim final action with comment period.",
               response_evidence=ResponseEvidence.STRONG,
               response_header_matched=False)
    status, reason = derive_status([ifc], agenda(final_rule_undetermined=True), ALL_CLEAN)
    assert status is Status.UNKNOWN and reason is None


def test_failed_channel_forces_unknown():
    channels = dict(ALL_CLEAN)
    channels[Channel.FULLTEXT_SEARCH] = "failed:HTTP 503"
    status, reason = derive_status([], agenda(final_rule_undetermined=True), channels)
    assert status is Status.UNKNOWN and reason is None


def test_zero_result_channel_is_clean_not_a_failure():
    assert all_channels_clean(ALL_CLEAN) is True
    assert all_channels_clean({**ALL_CLEAN, Channel.DOCKET_SEARCH: "skipped:no docket"}) is False


def test_missing_agenda_forces_unknown():
    status, reason = derive_status([], None, ALL_CLEAN)
    assert status is Status.UNKNOWN and reason is None
    status, reason = derive_status([], agenda(ok=False), ALL_CLEAN)
    assert status is Status.UNKNOWN and reason is None


def test_blm_direct_final_is_no_venue_possible():
    dfr = cand(rule_class=RuleClass.DIRECT_FINAL, action="Direct final rule.",
               postdates_comment=False, response_evidence=ResponseEvidence.NONE)
    status, reason = derive_status([dfr], agenda(stage="COMPLETED"), ALL_CLEAN)
    assert status is Status.CONFIDENTLY_ABSENT
    assert reason is AbsenceReason.NO_VENUE_POSSIBLE


def test_direct_final_carrying_real_responses_is_not_no_venue_possible():
    dfr = cand(rule_class=RuleClass.DIRECT_FINAL, action="Direct final rule.",
               postdates_comment=True, response_evidence=ResponseEvidence.STRONG)
    status, reason = derive_status([dfr], agenda(stage="COMPLETED"), ALL_CLEAN)
    assert status is Status.UNKNOWN and reason is None


def test_withdrawn_direct_final_is_not_no_venue_possible():
    dfr = cand(rule_class=RuleClass.DIRECT_FINAL, postdates_comment=False,
               response_evidence=ResponseEvidence.NONE)
    status, reason = derive_status([dfr], agenda(withdrawn=True, stage="COMPLETED"), ALL_CLEAN)
    assert status is not Status.CONFIDENTLY_ABSENT or reason is not AbsenceReason.NO_VENUE_POSSIBLE


def test_fbi_nprm_only_is_response_not_yet_published():
    nprm = cand(rule_class=RuleClass.PROPOSED,
                action="Notice of proposed rulemaking (NPRM).",
                postdates_comment=False, response_evidence=ResponseEvidence.NONE)
    status, reason = derive_status([nprm], agenda(stage="FINAL"), ALL_CLEAN)
    assert status is Status.CONFIDENTLY_ABSENT
    assert reason is AbsenceReason.RESPONSE_NOT_YET_PUBLISHED


def test_epa_tbd_agenda_outranks_not_yet_published():
    nprm = cand(rule_class=RuleClass.PROPOSED, action="Proposed rule.",
                postdates_comment=False, response_evidence=ResponseEvidence.NONE)
    status, reason = derive_status(
        [nprm], agenda(stage="LONG_TERM", final_rule_undetermined=True), ALL_CLEAN)
    assert status is Status.CONFIDENTLY_ABSENT
    assert reason is AbsenceReason.NO_FINAL_RULE_PLANNED


def test_no_candidates_and_no_corroboration_is_unknown():
    status, reason = derive_status([], agenda(stage="FINAL"), ALL_CLEAN)
    assert status is Status.UNKNOWN and reason is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resolution_status.py -v`
Expected: FAIL — `ModuleNotFoundError: ...resolution.status`.

- [ ] **Step 3: Implement**

```python
# stratification_scripts/resolution/status.py
"""Three-valued status derivation.

Absence is not provable by a finite search, so the layer bounds and calibrates
instead of guaranteeing: results are reported relative to the five declared
channels, and UNKNOWN is never collapsed into absence.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Tuple

from .models import (
    AbsenceReason, AgendaStatus, CandidateDocument, Channel, Relevance,
    ResponseEvidence, RuleClass, Status,
)

CHANNEL_OK = "ok"

_NO_VENUE_CLASSES = {RuleClass.DIRECT_FINAL, RuleClass.CONFIRMATION_OF_EFFECTIVE_DATE}
_NOT_YET_CLASSES = {RuleClass.PROPOSED, RuleClass.INTERIM_FINAL}


def qualifies(candidate: CandidateDocument) -> bool:
    """A candidate that could carry the agency's response to this comment.

    Response evidence is deliberately NOT part of qualification — gating on it
    would disqualify DOT 2024-29990 (a hand-confirmed real response whose
    response header does not match) and manufacture a false absence.
    """
    return (
        candidate.rule_class is RuleClass.FINAL
        and candidate.postdates_comment
        and candidate.relevance is Relevance.MATCH
    )


def all_channels_clean(channels_run: Mapping[Channel, str]) -> bool:
    """True only when every declared channel ran without failure or skip.

    A channel that ran and returned zero documents is clean; a skip is not.
    """
    return all(channels_run.get(c) == CHANNEL_OK for c in Channel)


def _blocks_absence(candidates: Iterable[CandidateDocument]) -> bool:
    """A deferred-response variant that postdates the comment and carries real text.

    An interim final rule or a "final rule with request for comments" can answer
    earlier-stage comments in its own preamble. Keyed off grounded-text evidence,
    never off response_header_matched, so this check does not inherit the
    header-flag defect it exists alongside.
    """
    return any(
        c.rule_class is not RuleClass.FINAL
        and c.postdates_comment
        and c.relevance is Relevance.MATCH
        and c.response_evidence is ResponseEvidence.STRONG
        for c in candidates
    )


def _absence_reason(
    candidates: List[CandidateDocument], agenda: AgendaStatus
) -> Optional[AbsenceReason]:
    """Corroboration is per-reason, not one global agenda clause."""
    structural = [
        c for c in candidates
        if c.rule_class in _NO_VENUE_CLASSES
        and c.relevance is Relevance.MATCH
        and c.response_evidence is not ResponseEvidence.STRONG
    ]
    if structural and not agenda.withdrawn:
        return AbsenceReason.NO_VENUE_POSSIBLE

    if agenda.final_rule_undetermined or (agenda.stage or "").upper() == "LONG_TERM":
        return AbsenceReason.NO_FINAL_RULE_PLANNED

    pending = [
        c for c in candidates
        if c.rule_class in _NOT_YET_CLASSES and c.relevance is Relevance.MATCH
    ]
    if pending:
        return AbsenceReason.RESPONSE_NOT_YET_PUBLISHED

    return None


def derive_status(
    candidates: List[CandidateDocument],
    agenda: Optional[AgendaStatus],
    channels_run: Mapping[Channel, str],
) -> Tuple[Status, Optional[AbsenceReason]]:
    """Derive (status, absence_reason) from candidates + agenda + envelope."""
    qualifying = [c for c in candidates if qualifies(c)]

    # Presence is cheap: FOUND needs only its own evidence, not a full sweep.
    if any(c.response_evidence is not ResponseEvidence.NONE for c in qualifying):
        return Status.FOUND, None

    # Venue found but unreadable — we know less than an absence claim requires.
    if qualifying:
        return Status.UNKNOWN, None

    if _blocks_absence(candidates):
        return Status.UNKNOWN, None

    # Absence is the expensive assertion: it needs the whole envelope, clean.
    if not all_channels_clean(channels_run):
        return Status.UNKNOWN, None
    if agenda is None or not agenda.ok:
        return Status.UNKNOWN, None

    reason = _absence_reason(candidates, agenda)
    if reason is None:
        return Status.UNKNOWN, None
    return Status.CONFIDENTLY_ABSENT, reason
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_resolution_status.py -v`
Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/resolution/status.py tests/test_resolution_status.py
git commit -m "feat(resolution): three-valued status + per-reason absence corroboration"
```

---

### Task 7: Federal Register search methods

**Files:**
- Modify: `stratification_scripts/federal_register/client.py` (add methods to `FederalRegisterClient`, after `lookup_document_by_citation`)
- Test: `tests/test_fr_client_search.py`

**Interfaces — Produces (on `FederalRegisterClient`):**
- `SEARCH_FIELDS: List[str]` — module constant
- `search_documents(self, conditions: Dict[str, str], *, per_page: int = 1000) -> Optional[List[dict]]` — returns `[]` for a clean zero-result query, `None` on failure
- `search_by_rin(self, rin: str) -> Optional[List[dict]]`
- `search_by_docket(self, docket_id: str) -> Optional[List[dict]]`
- `search_full_text(self, identifier: str) -> Optional[List[dict]]` — wraps the identifier in double quotes

**`None` vs `[]` matters:** `[]` is a clean run (`channels_run == "ok"`), `None` is a failure (`"failed:…"`) and forces `UNKNOWN` on any absence claim. Never conflate them.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fr_client_search.py
import json
from unittest.mock import MagicMock

from stratification_scripts.federal_register.client import (
    SEARCH_FIELDS, FederalRegisterClient,
)


def _client():
    return FederalRegisterClient(max_retries=1, sleep_between=0)


def _ok(payload):
    r = MagicMock(status_code=200)
    r.json.return_value = payload
    return r


def test_search_by_rin_sends_the_rin_condition_and_fields(monkeypatch):
    captured = {}

    def fake_get(url, params=None, timeout=30):
        captured["url"] = url
        captured["params"] = params
        return _ok({"count": 1, "results": [{"document_number": "2024-27333",
                                             "action": "Direct final rule."}]})

    c = _client()
    monkeypatch.setattr(c.session, "get", fake_get)
    docs = c.search_by_rin("1004-AF01")
    assert docs == [{"document_number": "2024-27333", "action": "Direct final rule."}]
    assert captured["params"]["conditions[regulation_id_number]"] == "1004-AF01"
    assert captured["params"]["fields[]"] == SEARCH_FIELDS


def test_search_by_docket_zero_results_is_empty_list_not_none(monkeypatch):
    c = _client()
    monkeypatch.setattr(c.session, "get",
                        lambda url, params=None, timeout=30: _ok({"count": 0}))
    assert c.search_by_docket("NOAA-NMFS-2023-0125") == []


def test_search_full_text_quotes_the_identifier(monkeypatch):
    captured = {}

    def fake_get(url, params=None, timeout=30):
        captured["params"] = params
        return _ok({"count": 0})

    c = _client()
    monkeypatch.setattr(c.session, "get", fake_get)
    c.search_full_text("1808-IFC")
    assert captured["params"]["conditions[term]"] == '"1808-IFC"'


def test_search_returns_none_on_persistent_http_error(monkeypatch):
    c = _client()
    monkeypatch.setattr(c.session, "get",
                        lambda url, params=None, timeout=30: MagicMock(status_code=503))
    assert c.search_by_rin("1004-AF01") is None


def test_search_returns_none_on_request_exception(monkeypatch):
    import requests

    def boom(url, params=None, timeout=30):
        raise requests.RequestException("network down")

    c = _client()
    monkeypatch.setattr(c.session, "get", boom)
    assert c.search_full_text("1808-IFC") is None


def test_blank_query_short_circuits_to_empty(monkeypatch):
    c = _client()
    monkeypatch.setattr(c.session, "get",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no request")))
    assert c.search_by_rin("") == []
    assert c.search_by_docket(None) == []
    assert c.search_full_text("  ") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fr_client_search.py -v`
Expected: FAIL — `ImportError: cannot import name 'SEARCH_FIELDS'`.

- [ ] **Step 3: Implement**

Add the constant next to `FR_DOCS_URL` in `stratification_scripts/federal_register/client.py`:

```python
# Fields requested by the resolution-layer search channels. Verified live against
# the FR API on 2026-07-23; `agencies` may contain entries whose name is null.
SEARCH_FIELDS = [
    "document_number", "title", "type", "action", "publication_date",
    "agencies", "regulation_id_numbers", "docket_ids", "citation",
]
```

Add these methods to `FederalRegisterClient` after `lookup_document_by_citation`:

```python
    def search_documents(
        self,
        conditions: Dict[str, str],
        *,
        per_page: int = 1000,
    ) -> Optional[List[dict]]:
        """Run one FR documents.json search.

        Returns the result list ([] for a clean zero-result query) or None if the
        request failed. The distinction is load-bearing downstream: [] is a clean
        channel run, None forces UNKNOWN on any absence claim.
        """
        if not conditions or not all(str(v).strip() for v in conditions.values()):
            return []

        if self.sleep_between > 0:
            time.sleep(self.sleep_between)

        params: Dict[str, Any] = {"per_page": per_page, "page": 1,
                                  "fields[]": SEARCH_FIELDS}
        params.update(conditions)

        backoff = 1.0
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(FR_DOCS_URL, params=params, timeout=30)
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    logger.warning(f"FR search failed for {conditions}: {type(e).__name__}: {e}")
                    return None
                time.sleep(backoff)
                backoff = min(backoff * 2, 16)
                continue

            if r.status_code == 200:
                data = r.json() or {}
                return list(data.get("results") or [])
            if r.status_code == 404:
                return []
            if r.status_code in (403, 429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                time.sleep(backoff)
                backoff = min(backoff * 2, 16)
                continue
            logger.warning(f"FR search HTTP {r.status_code} for {conditions}")
            return None
        return None

    def search_by_rin(self, rin: str) -> Optional[List[dict]]:
        """All FR documents filed under a RIN."""
        return self.search_documents({"conditions[regulation_id_number]": (rin or "").strip()})

    def search_by_docket(self, docket_id: Optional[str]) -> Optional[List[dict]]:
        """All FR documents filed under an agency docket id."""
        return self.search_documents({"conditions[docket_id]": (docket_id or "").strip()})

    def search_full_text(self, identifier: str) -> Optional[List[dict]]:
        """Full-text search on an IDENTIFIER — never a subject term.

        Measured: "Method 320" returns 83 unrelated rules that merely cite the
        method; "1808-IFC" returns 3, one of them exactly right. Identifiers give
        precision; topics give noise.
        """
        term = (identifier or "").strip()
        if not term:
            return []
        return self.search_documents({"conditions[term]": f'"{term}"'})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fr_client_search.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/federal_register/client.py tests/test_fr_client_search.py
git commit -m "feat(fr): documents.json search by RIN, docket, and quoted identifier"
```

---

### Task 8: Cross-row document cache

**Files:**
- Create: `stratification_scripts/resolution/cache.py`
- Test: `tests/test_resolution_cache.py`

**Interfaces:**
- Consumes: `FederalRegisterClient` (Task 7)
- Produces:
```python
class DocumentCache:
    def __init__(self, fr_client) -> None
    def details(self, document_number: str) -> Optional[dict]
    def xml(self, document_number: str) -> Optional[str]
    def extract(self, document_number: str) -> Optional[ResponseExtract]
    @property
    def stats(self) -> dict            # {"details_hits", "details_misses", "xml_hits", "xml_misses"}
```

**Why:** many comments on the same rulemaking share the same final rule; without a cross-row cache the same 5 MB XML is fetched once per comment. One observed umbrella RIN (FAA) returns 100 documents.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_resolution_cache.py
from stratification_scripts.resolution.cache import DocumentCache


class FakeFR:
    def __init__(self):
        self.detail_calls = []
        self.xml_calls = []

    def fetch_document_details(self, document_number, enrich_identifiers=True):
        self.detail_calls.append(document_number)
        return {"document_number": document_number, "action": "Final rule."}

    def fetch_document_full_text_xml(self, document_number):
        self.xml_calls.append(document_number)
        return "<RULE><SUPLINF>Comment: x Response: we agree</SUPLINF></RULE>"


def test_details_fetched_once_per_document():
    fr = FakeFR()
    cache = DocumentCache(fr)
    assert cache.details("2024-15931")["action"] == "Final rule."
    assert cache.details("2024-15931")["action"] == "Final rule."
    assert fr.detail_calls == ["2024-15931"]
    assert cache.stats["details_hits"] == 1


def test_xml_and_extract_share_one_fetch():
    fr = FakeFR()
    cache = DocumentCache(fr)
    assert "SUPLINF" in cache.xml("2024-29990")
    extract = cache.extract("2024-29990")
    assert extract is not None and extract.grounded_text
    cache.extract("2024-29990")
    assert fr.xml_calls == ["2024-29990"]


def test_failed_fetch_is_cached_as_a_miss_not_retried():
    class FailingFR(FakeFR):
        def fetch_document_full_text_xml(self, document_number):
            self.xml_calls.append(document_number)
            return None

    fr = FailingFR()
    cache = DocumentCache(fr)
    assert cache.xml("2024-00000") is None
    assert cache.xml("2024-00000") is None
    assert fr.xml_calls == ["2024-00000"]


def test_blank_document_number_never_hits_the_client():
    fr = FakeFR()
    cache = DocumentCache(fr)
    assert cache.details("") is None
    assert cache.xml(None) is None
    assert fr.detail_calls == [] and fr.xml_calls == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resolution_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: ...resolution.cache`.

- [ ] **Step 3: Implement**

```python
# stratification_scripts/resolution/cache.py
"""Cross-row document cache.

Many comments on the same rulemaking resolve to the same final rule, and one
observed umbrella RIN returns 100 documents. Without a cache shared across rows,
the same multi-megabyte XML would be fetched once per comment.
"""

from __future__ import annotations

from typing import Dict, Optional

from ..makeup.fr_response_extractor import ResponseExtract, extract_response_section

_MISS = object()


class DocumentCache:
    """Memoizes FR document details, full-text XML, and the response extract."""

    def __init__(self, fr_client) -> None:
        self._fr = fr_client
        self._details: Dict[str, object] = {}
        self._xml: Dict[str, object] = {}
        self._extract: Dict[str, object] = {}
        self._stats = {"details_hits": 0, "details_misses": 0,
                       "xml_hits": 0, "xml_misses": 0}

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def details(self, document_number: Optional[str]) -> Optional[dict]:
        key = (document_number or "").strip()
        if not key:
            return None
        if key in self._details:
            self._stats["details_hits"] += 1
            value = self._details[key]
            return None if value is _MISS else value
        self._stats["details_misses"] += 1
        fetched = self._fr.fetch_document_details(key, enrich_identifiers=True)
        self._details[key] = fetched if fetched else _MISS
        return fetched or None

    def xml(self, document_number: Optional[str]) -> Optional[str]:
        key = (document_number or "").strip()
        if not key:
            return None
        if key in self._xml:
            self._stats["xml_hits"] += 1
            value = self._xml[key]
            return None if value is _MISS else value
        self._stats["xml_misses"] += 1
        fetched = self._fr.fetch_document_full_text_xml(key)
        self._xml[key] = fetched if fetched else _MISS
        return fetched or None

    def extract(self, document_number: Optional[str]) -> Optional[ResponseExtract]:
        key = (document_number or "").strip()
        if not key:
            return None
        if key not in self._extract:
            xml = self.xml(key)
            self._extract[key] = extract_response_section(xml) if xml else _MISS
        value = self._extract[key]
        return None if value is _MISS else value
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_resolution_cache.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/resolution/cache.py tests/test_resolution_cache.py
git commit -m "feat(resolution): cross-row document + extract cache"
```

---

### Task 9: Discovery channels 1–3 (packet link, RIN, docket)

**Files:**
- Create: `stratification_scripts/resolution/channels.py`
- Test: `tests/test_resolution_channels.py`

**Interfaces:**
- Consumes: `CommentRef`, `CandidateDocument`, `Channel`, `Relevance` (Task 2); `rule_class_from_action` (Task 3); `postdates_comment`, `relevance_of` (Task 4); `DocumentCache` (Task 8); `search_by_rin`, `search_by_docket` (Task 7)
- Produces:
```python
ChannelOutcome = namedtuple-like dataclass: candidates: List[CandidateDocument], state: str
def candidate_from_fr_doc(doc: dict, *, ref: CommentRef, discovered_by: Channel) -> CandidateDocument
def run_packet_link(ref, cache) -> ChannelOutcome
def run_rin_search(ref, fr_client) -> ChannelOutcome
def run_docket_search(ref, fr_client) -> ChannelOutcome
```
`state` is `"ok"`, `"skipped:<why>"`, or `"failed:<why>"` — fed straight into `channels_run`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_resolution_channels.py
from stratification_scripts.resolution.cache import DocumentCache
from stratification_scripts.resolution.channels import (
    candidate_from_fr_doc, run_docket_search, run_packet_link, run_rin_search,
)
from stratification_scripts.resolution.models import (
    Channel, CommentRef, Relevance, RuleClass,
)

DOT_REF = CommentRef(
    comment_id="DOT-OST-2024-0090-0049", comment_date="2024-09-23",
    source_document="2024-18496", agency="Transportation Department",
    rins=("2105-AF05",), docket_id="Docket No. DOT-OST-2024-0090",
    packet_final_document="2025-02747",
)

FCC_DOC = {
    "document_number": "2025-02747", "title": "Radio Broadcasting Services",
    "type": "Rule", "action": "Final rule.", "publication_date": "2025-02-19",
    "agencies": [{"name": "Federal Communications Commission"}],
    "regulation_id_numbers": [], "docket_ids": ["DA 25-120"],
}

DOT_FINAL = {
    "document_number": "2024-29990", "title": "Transportation for Individuals",
    "type": "Rule", "action": "Final rule.", "publication_date": "2024-12-18",
    "agencies": [{"name": "Transportation Department"}, {"name": None}],
    "regulation_id_numbers": ["2105-AF05"], "docket_ids": ["Docket No. DOT-OST-2024-0090"],
}

DOT_NPRM = {
    "document_number": "2024-18496", "title": "NPRM", "type": "Proposed Rule",
    "action": "Notice of proposed rulemaking (NPRM).", "publication_date": "2024-08-22",
    "agencies": [{"name": "Transportation Department"}],
    "regulation_id_numbers": ["2105-AF05"], "docket_ids": ["Docket No. DOT-OST-2024-0090"],
}


class FakeFR:
    def __init__(self, rin_docs=None, docket_docs=None, details=None):
        self._rin_docs, self._docket_docs = rin_docs, docket_docs
        self._details = details or {}

    def search_by_rin(self, rin):
        return self._rin_docs

    def search_by_docket(self, docket_id):
        return self._docket_docs

    def fetch_document_details(self, document_number, enrich_identifiers=True):
        return self._details.get(document_number)

    def fetch_document_full_text_xml(self, document_number):
        return None


def test_candidate_carries_provenance_and_classification():
    c = candidate_from_fr_doc(DOT_FINAL, ref=DOT_REF, discovered_by=Channel.RIN_SEARCH)
    assert c.document_number == "2024-29990"
    assert c.rule_class is RuleClass.FINAL
    assert c.discovered_by is Channel.RIN_SEARCH
    assert c.postdates_comment is True
    assert c.relevance is Relevance.MATCH
    assert c.agency_names == ("Transportation Department",)   # None entries dropped


def test_packet_link_to_another_agency_is_returned_but_marked_mismatch():
    # The candidate is NOT discarded — it explains the resolution failure.
    cache = DocumentCache(FakeFR(details={"2025-02747": FCC_DOC}))
    outcome = run_packet_link(DOT_REF, cache)
    assert outcome.state == "ok"
    assert [c.document_number for c in outcome.candidates] == ["2025-02747"]
    assert outcome.candidates[0].relevance is Relevance.AGENCY_MISMATCH


def test_packet_link_absent_is_skipped_not_failed():
    ref = CommentRef(comment_id="x", comment_date="2024-01-01", source_document="d",
                     agency="A", rins=(), docket_id=None, packet_final_document=None)
    outcome = run_packet_link(ref, DocumentCache(FakeFR()))
    assert outcome.candidates == []
    assert outcome.state.startswith("skipped:")


def test_packet_link_fetch_failure_is_failed():
    outcome = run_packet_link(DOT_REF, DocumentCache(FakeFR(details={})))
    assert outcome.state.startswith("failed:")


def test_rin_search_returns_all_docs_under_the_rin():
    fr = FakeFR(rin_docs=[DOT_FINAL, DOT_NPRM])
    outcome = run_rin_search(DOT_REF, fr)
    assert outcome.state == "ok"
    assert {c.document_number for c in outcome.candidates} == {"2024-29990", "2024-18496"}
    by_num = {c.document_number: c for c in outcome.candidates}
    assert by_num["2024-18496"].rule_class is RuleClass.PROPOSED
    assert by_num["2024-18496"].postdates_comment is False


def test_rin_search_failure_is_failed_not_empty():
    outcome = run_rin_search(DOT_REF, FakeFR(rin_docs=None))
    assert outcome.candidates == [] and outcome.state.startswith("failed:")


def test_docket_search_zero_results_is_ok():
    outcome = run_docket_search(DOT_REF, FakeFR(docket_docs=[]))
    assert outcome.candidates == [] and outcome.state == "ok"


def test_docket_search_without_a_docket_is_skipped():
    ref = CommentRef(comment_id="x", comment_date="2024-01-01", source_document="d",
                     agency="A", rins=("1234-AB56",), docket_id=None,
                     packet_final_document=None)
    outcome = run_docket_search(ref, FakeFR(docket_docs=[]))
    assert outcome.state.startswith("skipped:")


def test_multi_rin_search_is_deduplicated():
    ref = CommentRef(comment_id="x", comment_date="2024-01-01", source_document="d",
                     agency="Transportation Department",
                     rins=("2105-AF05", "2105-AF05"), docket_id=None,
                     packet_final_document=None)
    outcome = run_rin_search(ref, FakeFR(rin_docs=[DOT_FINAL]))
    assert len(outcome.candidates) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resolution_channels.py -v`
Expected: FAIL — `ModuleNotFoundError: ...resolution.channels`.

- [ ] **Step 3: Implement**

```python
# stratification_scripts/resolution/channels.py
"""The five discovery channels, ordered precise -> wide.

Each channel returns candidates plus a state string that goes straight into
ResolutionResult.channels_run. The distinction between "ran and found nothing"
and "did not run" is load-bearing: only the former can support an absence claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .classify import rule_class_from_action
from .filters import postdates_comment, relevance_of
from .models import CandidateDocument, Channel, CommentRef

STATE_OK = "ok"


@dataclass
class ChannelOutcome:
    candidates: List[CandidateDocument] = field(default_factory=list)
    state: str = STATE_OK


def _agency_names(doc: dict) -> Sequence[str]:
    names = []
    for agency in doc.get("agencies") or []:
        if isinstance(agency, dict):
            name = agency.get("name") or agency.get("raw_name")
        else:
            name = agency
        if name:
            names.append(str(name))
    return tuple(names)


def candidate_from_fr_doc(
    doc: dict, *, ref: CommentRef, discovered_by: Channel
) -> CandidateDocument:
    """Build a CandidateDocument from one FR API document record."""
    rins = tuple(str(r) for r in (doc.get("regulation_id_numbers") or []) if r)
    dockets = [str(d) for d in (doc.get("docket_ids") or []) if d]
    docket_id = dockets[0] if dockets else doc.get("docket_id")
    agency_names = _agency_names(doc)
    publication_date = doc.get("publication_date")
    return CandidateDocument(
        document_number=str(doc.get("document_number") or ""),
        publication_date=publication_date,
        type=doc.get("type"),
        action=doc.get("action"),
        title=doc.get("title"),
        agency_names=tuple(agency_names),
        rule_class=rule_class_from_action(doc.get("action"), doc.get("type")),
        rins=rins,
        docket_id=docket_id,
        discovered_by=discovered_by,
        postdates_comment=postdates_comment(publication_date, ref.comment_date),
        relevance=relevance_of(
            discovered_by=discovered_by, agency_names=agency_names,
            rins=rins, docket_id=docket_id, ref=ref,
        ),
    )


def run_packet_link(ref: CommentRef, cache) -> ChannelOutcome:
    """Channel 1 — the existing final_rule_document_number, trusted no further.

    The candidate is returned even when the relevance check rejects it: a
    rejected link EXPLAINS a resolution failure and must stay visible.
    """
    if not ref.packet_final_document:
        return ChannelOutcome([], "skipped:no packet link")
    details = cache.details(ref.packet_final_document)
    if not details:
        return ChannelOutcome([], f"failed:details fetch {ref.packet_final_document}")
    return ChannelOutcome(
        [candidate_from_fr_doc(details, ref=ref, discovered_by=Channel.PACKET_LINK)],
        STATE_OK,
    )


def _collect(
    docs: Optional[List[dict]], ref: CommentRef, channel: Channel
) -> List[CandidateDocument]:
    seen = set()
    out: List[CandidateDocument] = []
    for doc in docs or []:
        number = str(doc.get("document_number") or "")
        if not number or number in seen:
            continue
        seen.add(number)
        out.append(candidate_from_fr_doc(doc, ref=ref, discovered_by=channel))
    return out


def run_rin_search(ref: CommentRef, fr_client) -> ChannelOutcome:
    """Channel 2 — every FR document filed under any of the comment's RINs."""
    rins = [r for r in dict.fromkeys(ref.rins) if r]
    if not rins:
        return ChannelOutcome([], "skipped:no rin")
    docs: List[dict] = []
    for rin in rins:
        found = fr_client.search_by_rin(rin)
        if found is None:
            return ChannelOutcome([], f"failed:rin search {rin}")
        docs.extend(found)
    return ChannelOutcome(_collect(docs, ref, Channel.RIN_SEARCH), STATE_OK)


def run_docket_search(ref: CommentRef, fr_client) -> ChannelOutcome:
    """Channel 3 — every FR document filed under the comment's docket id.

    A zero-result docket query is a CLEAN run: many agencies (NOAA) file under a
    docket string the FR API does not index.
    """
    if not ref.docket_id:
        return ChannelOutcome([], "skipped:no docket")
    docs = fr_client.search_by_docket(ref.docket_id)
    if docs is None:
        return ChannelOutcome([], f"failed:docket search {ref.docket_id}")
    return ChannelOutcome(_collect(docs, ref, Channel.DOCKET_SEARCH), STATE_OK)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_resolution_channels.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/resolution/channels.py tests/test_resolution_channels.py
git commit -m "feat(resolution): channels 1-3 (packet link, RIN, docket) with typed states"
```

---

### Task 10: Resolver composition, fetch policy, and merge

**Files:**
- Create: `stratification_scripts/resolution/resolver.py`
- Test: `tests/test_resolution_resolver.py`

**Interfaces:**
- Consumes: everything from Tasks 2–9
- Produces:
```python
FETCHABLE_CLASSES: set[RuleClass]      # FINAL, INTERIM_FINAL, DIRECT_FINAL, CONFIRMATION_OF_EFFECTIVE_DATE
class DocumentResolver:
    def __init__(self, *, fr_client, reginfo_client=None, cache=None) -> None
    def resolve(self, ref: CommentRef) -> ResolutionResult
def merge_candidates(groups: List[List[CandidateDocument]]) -> List[CandidateDocument]
def qualifying_candidates(result: ResolutionResult) -> List[CandidateDocument]
```

**Fetch policy (cost control):** XML is fetched only for candidates that survive chronology + relevance + rule-class filters — `relevance == MATCH ∧ postdates_comment ∧ rule_class ∈ FETCHABLE_CLASSES`. `DIRECT_FINAL` and `CONFIRMATION_OF_EFFECTIVE_DATE` are fetchable because a `NO_VENUE_POSSIBLE` claim rests on them carrying no real responses; `PROPOSED` and `OTHER` are never fetched.

**Merge:** the same document can surface from several channels. Keep one candidate per `document_number`, retaining the *most precise* channel that found it (`PACKET_LINK < RIN_SEARCH < DOCKET_SEARCH < FULLTEXT_SEARCH`), and re-run `relevance_of` under the retained channel so a document found by both a wide channel and the packet link is not falsely rescued.

**Chosen-candidate policy for consumers:** `qualifying_candidates()` returns **all** qualifying candidates, earliest-publication-first. Do not pre-select one — a final rule plus its correction, or two finals under different RINs, are all plausibly the venue, and judgment is already reading text. If a caller needs exactly one, the documented default is the first element.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_resolution_resolver.py
from stratification_scripts.resolution.models import (
    CandidateDocument, Channel, CommentRef, Relevance, ResponseEvidence,
    RuleClass, Status,
)
from stratification_scripts.resolution.resolver import (
    DocumentResolver, merge_candidates, qualifying_candidates,
)

REF = CommentRef(
    comment_id="DOT-OST-2024-0090-0049", comment_date="2024-09-23",
    source_document="2024-18496", agency="Transportation Department",
    rins=("2105-AF05",), docket_id="Docket No. DOT-OST-2024-0090",
    packet_final_document="2025-02747",
)

FCC_DOC = {"document_number": "2025-02747", "title": "Radio Broadcasting Services",
           "type": "Rule", "action": "Final rule.", "publication_date": "2025-02-19",
           "agencies": [{"name": "Federal Communications Commission"}],
           "regulation_id_numbers": [], "docket_ids": ["DA 25-120"]}

DOT_FINAL = {"document_number": "2024-29990", "title": "T", "type": "Rule",
             "action": "Final rule.", "publication_date": "2024-12-18",
             "agencies": [{"name": "Transportation Department"}, {"name": None}],
             "regulation_id_numbers": ["2105-AF05"],
             "docket_ids": ["Docket No. DOT-OST-2024-0090"]}

DOT_NPRM = {"document_number": "2024-18496", "title": "N", "type": "Proposed Rule",
            "action": "Notice of proposed rulemaking (NPRM).",
            "publication_date": "2024-08-22",
            "agencies": [{"name": "Transportation Department"}],
            "regulation_id_numbers": ["2105-AF05"],
            "docket_ids": ["Docket No. DOT-OST-2024-0090"]}

DENSE_XML = (
    "<RULE><SUPLINF>"
    + ("Comment: Several commenters argued this. Response: We disagree with the "
       "commenters and are adopting the provision. In response to comment, the "
       "agency considered it. ") * 20
    + "</SUPLINF></RULE>"
)


class FakeFR:
    def __init__(self):
        self.xml_calls = []
        self.docs = {d["document_number"]: d for d in (FCC_DOC, DOT_FINAL, DOT_NPRM)}

    def fetch_document_details(self, document_number, enrich_identifiers=True):
        return self.docs.get(document_number)

    def fetch_document_full_text_xml(self, document_number):
        self.xml_calls.append(document_number)
        return DENSE_XML

    def search_by_rin(self, rin):
        return [DOT_FINAL, DOT_NPRM]

    def search_by_docket(self, docket_id):
        return []

    def search_full_text(self, identifier):
        return []


class FakeRegInfo:
    def fetch_unified_agenda(self, rin):
        return {"rin": rin, "stage": "FINAL", "timetable": [
            {"action": "NPRM", "date": "2024-08-22", "date_raw": "", "citation": ""},
            {"action": "FINAL RULE", "date": "2024-12-18", "date_raw": "", "citation": ""},
        ], "withdrawn": False}


def test_dot_row_resolves_found_via_rin_search_despite_bad_packet_link():
    fr = FakeFR()
    result = DocumentResolver(fr_client=fr, reginfo_client=FakeRegInfo()).resolve(REF)
    assert result.status is Status.FOUND
    assert result.absence_reason is None
    by_num = {c.document_number: c for c in result.candidates}
    assert by_num["2025-02747"].relevance is Relevance.AGENCY_MISMATCH
    assert by_num["2024-29990"].discovered_by is Channel.RIN_SEARCH
    assert by_num["2024-29990"].response_evidence is ResponseEvidence.STRONG


def test_fetch_policy_skips_non_qualifying_candidates():
    fr = FakeFR()
    DocumentResolver(fr_client=fr, reginfo_client=FakeRegInfo()).resolve(REF)
    # Only the DOT final rule survives chronology + relevance + rule class.
    assert fr.xml_calls == ["2024-29990"]


def test_channels_run_records_the_envelope_per_row():
    result = DocumentResolver(fr_client=FakeFR(), reginfo_client=FakeRegInfo()).resolve(REF)
    assert set(result.channels_run) == set(Channel)
    assert result.channels_run[Channel.DOCKET_SEARCH] == "ok"     # zero results, clean


def test_merge_keeps_the_most_precise_channel():
    def make(channel):
        return CandidateDocument(
            document_number="2024-29990", publication_date="2024-12-18", type="Rule",
            action="Final rule.", title="t", agency_names=("Transportation Department",),
            rule_class=RuleClass.FINAL, rins=("2105-AF05",), docket_id=None,
            discovered_by=channel, postdates_comment=True, relevance=Relevance.MATCH,
        )

    merged = merge_candidates([[make(Channel.FULLTEXT_SEARCH)], [make(Channel.RIN_SEARCH)]])
    assert len(merged) == 1
    assert merged[0].discovered_by is Channel.RIN_SEARCH


def test_qualifying_candidates_are_all_returned_earliest_first():
    result = DocumentResolver(fr_client=FakeFR(), reginfo_client=FakeRegInfo()).resolve(REF)
    quals = qualifying_candidates(result)
    assert [c.document_number for c in quals] == ["2024-29990"]


def test_agenda_failure_is_recorded_and_forces_unknown_on_absence():
    class NoAgenda(FakeRegInfo):
        def fetch_unified_agenda(self, rin):
            return None

    class EmptyFR(FakeFR):
        def fetch_document_details(self, document_number, enrich_identifiers=True):
            return None

        def search_by_rin(self, rin):
            return []

    result = DocumentResolver(fr_client=EmptyFR(), reginfo_client=NoAgenda()).resolve(REF)
    assert result.status is Status.UNKNOWN
    assert result.agenda is not None and result.agenda.ok is False
    assert result.channels_run[Channel.AGENDA].startswith("failed:")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resolution_resolver.py -v`
Expected: FAIL — `ModuleNotFoundError: ...resolution.resolver`.

- [ ] **Step 3: Implement**

```python
# stratification_scripts/resolution/resolver.py
"""Composition: run the channels, apply the fetch policy, derive the status.

The resolver takes injected clients so the acceptance suite can replay recorded
API responses offline.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from ..reginfo.client import has_undetermined_final_rule
from .cache import DocumentCache
from .channels import (
    ChannelOutcome, run_docket_search, run_packet_link, run_rin_search,
)
from .evidence import response_evidence_from_extract
from .filters import relevance_of
from .models import (
    AgendaStatus, CandidateDocument, Channel, CommentRef, Relevance,
    ResolutionResult, RuleClass, Status,
)
from .status import derive_status, qualifies

# Only these classes are worth an XML fetch: FINAL because it is the venue,
# INTERIM_FINAL because it can answer earlier-stage comments, and the two
# structural classes because a NO_VENUE_POSSIBLE claim rests on them being empty.
FETCHABLE_CLASSES = {
    RuleClass.FINAL,
    RuleClass.INTERIM_FINAL,
    RuleClass.DIRECT_FINAL,
    RuleClass.CONFIRMATION_OF_EFFECTIVE_DATE,
}

# Precise -> wide. On a tie the more precise channel keeps the candidate.
_CHANNEL_PRECEDENCE = {
    Channel.PACKET_LINK: 0,
    Channel.RIN_SEARCH: 1,
    Channel.DOCKET_SEARCH: 2,
    Channel.AGENDA: 3,
    Channel.FULLTEXT_SEARCH: 4,
}


def merge_candidates(groups: List[List[CandidateDocument]]) -> List[CandidateDocument]:
    """One candidate per document number, keeping the most precise channel."""
    best: Dict[str, CandidateDocument] = {}
    for group in groups:
        for candidate in group:
            existing = best.get(candidate.document_number)
            if existing is None or (
                _CHANNEL_PRECEDENCE[candidate.discovered_by]
                < _CHANNEL_PRECEDENCE[existing.discovered_by]
            ):
                best[candidate.document_number] = candidate
    return sorted(best.values(), key=lambda c: (c.publication_date or "", c.document_number))


def qualifying_candidates(result: ResolutionResult) -> List[CandidateDocument]:
    """All qualifying candidates, earliest publication first.

    Consumers pass the WHOLE list to judgment. Pre-selecting one would reintroduce
    a fallible resolution decision; if a caller needs exactly one, the documented
    default is the first element.
    """
    return sorted(
        [c for c in result.candidates if qualifies(c)],
        key=lambda c: (c.publication_date or "", c.document_number),
    )


class DocumentResolver:
    """Given a comment, return every document where a response could live."""

    def __init__(self, *, fr_client, reginfo_client=None, cache=None) -> None:
        self._fr = fr_client
        self._reginfo = reginfo_client
        self._cache = cache or DocumentCache(fr_client)

    @property
    def cache(self) -> DocumentCache:
        return self._cache

    def _fetch_agenda(self, ref: CommentRef) -> tuple[AgendaStatus, str]:
        fetched_at = datetime.now().isoformat()
        rin = ref.rins[0] if ref.rins else None
        if not rin:
            return (AgendaStatus(None, None, [], False, False, fetched_at, False),
                    "skipped:no rin")
        if self._reginfo is None:
            return (AgendaStatus(rin, None, [], False, False, fetched_at, False),
                    "skipped:no reginfo client")
        try:
            agenda = self._reginfo.fetch_unified_agenda(rin)
        except Exception as exc:  # noqa: BLE001 — an agenda failure must not crash a run
            return (AgendaStatus(rin, None, [], False, False, fetched_at, False),
                    f"failed:agenda {type(exc).__name__}")
        if not agenda:
            return (AgendaStatus(rin, None, [], False, False, fetched_at, False),
                    "failed:agenda not found")
        return (
            AgendaStatus(
                rin=rin,
                stage=agenda.get("stage"),
                timetable=list(agenda.get("timetable") or []),
                final_rule_undetermined=has_undetermined_final_rule(agenda),
                withdrawn=bool(agenda.get("withdrawn")),
                fetched_at=fetched_at,
                ok=True,
            ),
            "ok",
        )

    def _run_fulltext(self, ref: CommentRef) -> ChannelOutcome:
        """Channel 5 — overridden in Task 11. Until then it is an honest skip."""
        return ChannelOutcome([], "skipped:not implemented")

    def _apply_fetch_policy(self, candidates: List[CandidateDocument]) -> None:
        """Attach graded response evidence to candidates worth reading."""
        for candidate in candidates:
            if candidate.relevance is not Relevance.MATCH:
                continue
            if not candidate.postdates_comment:
                continue
            if candidate.rule_class not in FETCHABLE_CLASSES:
                continue
            extract = self._cache.extract(candidate.document_number)
            candidate.response_evidence = response_evidence_from_extract(extract)
            if extract is not None:
                candidate.response_header_matched = extract.found_response_hd
                candidate.response_section_ref = extract.matched_header

    def resolve(self, ref: CommentRef) -> ResolutionResult:
        channels_run: Dict[Channel, str] = {}

        packet = run_packet_link(ref, self._cache)
        channels_run[Channel.PACKET_LINK] = packet.state

        rin = run_rin_search(ref, self._fr)
        channels_run[Channel.RIN_SEARCH] = rin.state

        docket = run_docket_search(ref, self._fr)
        channels_run[Channel.DOCKET_SEARCH] = docket.state

        agenda, agenda_state = self._fetch_agenda(ref)
        channels_run[Channel.AGENDA] = agenda_state

        fulltext = self._run_fulltext(ref)
        channels_run[Channel.FULLTEXT_SEARCH] = fulltext.state

        candidates = merge_candidates(
            [packet.candidates, rin.candidates, docket.candidates, fulltext.candidates]
        )
        # A merged candidate may have changed channels; re-derive its relevance so
        # a wide-channel hit is never rescued by a packet-link classification.
        for candidate in candidates:
            candidate.relevance = relevance_of(
                discovered_by=candidate.discovered_by,
                agency_names=candidate.agency_names,
                rins=candidate.rins,
                docket_id=candidate.docket_id,
                ref=ref,
            )

        self._apply_fetch_policy(candidates)
        status, reason = derive_status(candidates, agenda, channels_run)

        return ResolutionResult(
            comment_id=ref.comment_id,
            comment_date=ref.comment_date,
            source_document=ref.source_document,
            status=status,
            absence_reason=reason,
            candidates=candidates,
            agenda=agenda,
            channels_run=channels_run,
            resolved_at=datetime.now().isoformat(),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_resolution_resolver.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/resolution/resolver.py tests/test_resolution_resolver.py
git commit -m "feat(resolution): resolver composition, fetch policy, candidate merge"
```

---

### Task 11: Channel 5 — full-text identifier search

The channel that cracks the CMS cross-RIN topology, the one the current pipeline is structurally blind to.

**Files:**
- Modify: `stratification_scripts/resolution/channels.py` (add `run_fulltext_search`, `fulltext_identifiers`)
- Modify: `stratification_scripts/resolution/resolver.py` (replace the `_run_fulltext` stub)
- Test: `tests/test_resolution_channels.py`, `tests/test_resolution_resolver.py`

**Interfaces — Produces:**
- `fulltext_identifiers(ref: CommentRef) -> List[str]` — the docket id (normalized *and* raw, deduped) plus each RIN. Identifiers only, never subject terms.
- `run_fulltext_search(ref: CommentRef, fr_client) -> ChannelOutcome`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_resolution_channels.py`:

```python
from stratification_scripts.resolution.channels import (
    fulltext_identifiers, run_fulltext_search,
)

CMS_REF = CommentRef(
    comment_id="CMS-2024-0131-6043", comment_date="2024-12-03",
    source_document="2024-22765", agency="Health and Human Services Department",
    rins=("0938-AV34",), docket_id="CMS-1808-IFC", packet_final_document=None,
)

CMS_LATER = {
    "document_number": "2025-14681",
    "title": "Medicare Program; Hospital Inpatient Prospective Payment Systems",
    "type": "Rule", "action": "Final rule.", "publication_date": "2025-08-04",
    "agencies": [{"name": "Health and Human Services Department"},
                 {"name": "Centers for Medicare & Medicaid Services"}],
    "regulation_id_numbers": ["0938-AV53"], "docket_ids": ["CMS-1809-F"],
}


def test_fulltext_identifiers_are_identifiers_only():
    ids = fulltext_identifiers(CMS_REF)
    assert "CMS-1808-IFC" in ids
    assert "0938-AV34" in ids
    # No title/subject words ever leak into the query set.
    assert all(len(i.split()) == 1 for i in ids)


def test_fulltext_recovers_the_cross_rin_response():
    class FakeFullText:
        def __init__(self):
            self.terms = []

        def search_full_text(self, identifier):
            self.terms.append(identifier)
            return [CMS_LATER] if identifier == "CMS-1808-IFC" else []

    fr = FakeFullText()
    outcome = run_fulltext_search(CMS_REF, fr)
    assert outcome.state == "ok"
    assert [c.document_number for c in outcome.candidates] == ["2025-14681"]
    # Discovered under a DIFFERENT RIN and docket, and it must still be MATCH.
    assert outcome.candidates[0].relevance is Relevance.MATCH
    assert outcome.candidates[0].discovered_by is Channel.FULLTEXT_SEARCH


def test_fulltext_failure_on_any_identifier_is_failed():
    class Failing:
        def search_full_text(self, identifier):
            return None

    assert run_fulltext_search(CMS_REF, Failing()).state.startswith("failed:")


def test_fulltext_without_identifiers_is_skipped():
    ref = CommentRef(comment_id="x", comment_date="2024-01-01", source_document="d",
                     agency="A", rins=(), docket_id=None, packet_final_document=None)

    class Unused:
        def search_full_text(self, identifier):
            raise AssertionError("should not be called")

    assert run_fulltext_search(ref, Unused()).state.startswith("skipped:")
```

Append to `tests/test_resolution_resolver.py`:

```python
def test_resolver_runs_fulltext_and_marks_it_ok():
    result = DocumentResolver(fr_client=FakeFR(), reginfo_client=FakeRegInfo()).resolve(REF)
    assert result.channels_run[Channel.FULLTEXT_SEARCH] == "ok"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_resolution_channels.py tests/test_resolution_resolver.py -v`
Expected: FAIL — `ImportError: cannot import name 'fulltext_identifiers'`, and `test_resolver_runs_fulltext_and_marks_it_ok` fails with `"skipped:not implemented" != "ok"`.

- [ ] **Step 3: Implement**

Append to `stratification_scripts/resolution/channels.py`:

```python
from ..federal_register.client import normalize_docket_id


def fulltext_identifiers(ref: CommentRef) -> List[str]:
    """Identifier queries for channel 5 — never subject terms.

    Measured: "Method 320" returned 83 unrelated rules that merely cite the
    method; "1808-IFC" returned 3, one of them the document that answers the
    comment. Identifiers give precision; topics give noise.
    """
    identifiers: List[str] = []
    for value in [ref.docket_id, normalize_docket_id(ref.docket_id), *ref.rins]:
        token = (value or "").strip()
        if token and " " not in token and token not in identifiers:
            identifiers.append(token)
    return identifiers


def run_fulltext_search(ref: CommentRef, fr_client) -> ChannelOutcome:
    """Channel 5 — the wide net, keyed on identifiers.

    This is the only channel that can find a response published under a DIFFERENT
    RIN (CMS: the FY2026 IPPS final rule answering FY2025 IFC comments), a
    topology the packet-link model is structurally blind to.
    """
    identifiers = fulltext_identifiers(ref)
    if not identifiers:
        return ChannelOutcome([], "skipped:no identifiers")
    docs: List[dict] = []
    for identifier in identifiers:
        found = fr_client.search_full_text(identifier)
        if found is None:
            return ChannelOutcome([], f"failed:fulltext search {identifier}")
        docs.extend(found)
    return ChannelOutcome(_collect(docs, ref, Channel.FULLTEXT_SEARCH), STATE_OK)
```

In `stratification_scripts/resolution/resolver.py`, import `run_fulltext_search` alongside the other channels and replace the stub:

```python
    def _run_fulltext(self, ref: CommentRef) -> ChannelOutcome:
        return run_fulltext_search(ref, self._fr)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_resolution_channels.py tests/test_resolution_resolver.py -v`
Expected: 13 + 7 passed.

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/resolution/channels.py stratification_scripts/resolution/resolver.py tests/test_resolution_channels.py tests/test_resolution_resolver.py
git commit -m "feat(resolution): channel 5 full-text identifier search (cross-RIN recovery)"
```

---

### Task 12: The six-topology acceptance suite

Every fixture pins a distinct topology; together they are the regression suite for the whole ontology. This is the task the spec's acceptance criterion names.

**Files:**
- Create: `tests/fixtures/resolution/_record.py`, `tests/fixtures/resolution/README.md`, and the per-row fixture directories
- Create: `tests/test_resolution_acceptance.py`
- Test: `tests/test_resolution_acceptance.py`

**Interfaces:**
- Consumes: `DocumentResolver` (Tasks 10–11)
- Produces: `ReplayFRClient`, `ReplayRegInfoClient` (test-local), reading recorded JSON/XML from a fixture directory. No network in the test path.

**The six rows and their expected outcomes:**

| comment_id | topology | expected |
|---|---|---|
| `NOAA-NMFS-2023-0125-0016` | normal preamble | `FOUND`, `2024-15931` qualifying |
| `BLM-2024-0001-0003` | direct final rule | `CONFIDENTLY_ABSENT` / `NO_VENUE_POSSIBLE`; `2024-27333` returned, `rule_class=DIRECT_FINAL`, non-qualifying |
| `FBI-2024-0002-0006` | NPRM only | `CONFIDENTLY_ABSENT` / `RESPONSE_NOT_YET_PUBLISHED` |
| `EPA-HQ-OAR-2022-0491-0022` | agenda says To Be Determined | `CONFIDENTLY_ABSENT` / `NO_FINAL_RULE_PLANNED` |
| `DOT-OST-2024-0090-0049` | packet link → FCC; true final has no matching header | `FOUND` via `RIN_SEARCH` on `2024-29990`, `response_header_matched == False`; packet candidate `AGENCY_MISMATCH` |
| `CMS-2024-0131-6043` | response under a different RIN | `FOUND` via `FULLTEXT_SEARCH` on `2025-14681` |

**Fixture inputs** (from the frozen snapshot `2026-07-15-ce44ac5` and the goldset packet — already verified):

| comment_id | comment_date | source_document | agency | rin | docket_id | packet_final_document |
|---|---|---|---|---|---|---|
| `NOAA-NMFS-2023-0125-0016` | 2024-03-22 | 2024-01120 | Commerce Department, National Oceanic and Atmospheric Administration | 0648-BM40 | NOAA-NMFS-2023-0125 | 2024-15931 |
| `BLM-2024-0001-0003` | 2024-12-26 | 2024-27333 | Interior Department, Land Management Bureau | 1004-AF01 | BLM_HQ_FRN_MO4500181705 | 2024-27333 |
| `FBI-2024-0002-0006` | 2025-02-11 | 2024-28712 | Justice Department | 1110-AA36 | Docket No. FBI-158 | *(none)* |
| `EPA-HQ-OAR-2022-0491-0022` | 2024-05-15 | 2024-04359 | Environmental Protection Agency | 2060-AV81 | EPA-HQ-OAR-2022-0491 | *(none)* |
| `DOT-OST-2024-0090-0049` | 2024-09-23 | 2024-18496 | Transportation Department | 2105-AF05 | Docket No. DOT-OST-2024-0090 | 2025-02747 |
| `CMS-2024-0131-6043` | 2024-12-03 | 2024-22765 | Health and Human Services Department | 0938-AV34 | CMS-1808-IFC | *(none)* |

*(Use the exact `agency` strings from `frozen/2026-07-15-ce44ac5/output/federal_register_2024_comments.csv`; the recorder writes them into `input.json` so they are never retyped by hand.)*

- [ ] **Step 1: Write the recorder**

Create `tests/fixtures/resolution/_record.py` (run manually, never collected by pytest — the leading underscore keeps it out of discovery):

```python
"""Record live API responses for the six-topology acceptance fixtures.

Run manually:  python tests/fixtures/resolution/_record.py
Re-run only when a fixture's expected behavior genuinely changes; the committed
fixtures are what makes the acceptance suite deterministic and offline.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

from stratification_scripts.federal_register.client import (
    FederalRegisterClient, normalize_docket_id,
)
from stratification_scripts.reginfo.client import RegInfoClient
from stratification_scripts.resolution.channels import fulltext_identifiers
from stratification_scripts.resolution.models import CommentRef

HERE = Path(__file__).parent
MAX_XML_BYTES = 600_000     # above this, keep only the SUPLINF container

ROWS = [
    ("NOAA-NMFS-2023-0125-0016", "2024-03-22", "2024-01120",
     "Commerce Department, National Oceanic and Atmospheric Administration",
     ("0648-BM40",), "NOAA-NMFS-2023-0125", "2024-15931"),
    ("BLM-2024-0001-0003", "2024-12-26", "2024-27333",
     "Interior Department, Land Management Bureau",
     ("1004-AF01",), "BLM_HQ_FRN_MO4500181705", "2024-27333"),
    ("FBI-2024-0002-0006", "2025-02-11", "2024-28712", "Justice Department",
     ("1110-AA36",), "Docket No. FBI-158", None),
    ("EPA-HQ-OAR-2022-0491-0022", "2024-05-15", "2024-04359",
     "Environmental Protection Agency",
     ("2060-AV81",), "EPA-HQ-OAR-2022-0491", None),
    ("DOT-OST-2024-0090-0049", "2024-09-23", "2024-18496", "Transportation Department",
     ("2105-AF05",), "Docket No. DOT-OST-2024-0090", "2025-02747"),
    ("CMS-2024-0131-6043", "2024-12-03", "2024-22765",
     "Health and Human Services Department",
     ("0938-AV34",), "CMS-1808-IFC", None),
]


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def trim_xml(xml: str) -> str:
    """Keep the SUPLINF container for oversized documents (CMS is 5.5 MB)."""
    if len(xml.encode()) <= MAX_XML_BYTES:
        return xml
    match = re.search(r"<SUPLINF\b.*?</SUPLINF>", xml, re.S | re.I)
    body = match.group(0) if match else xml[:MAX_XML_BYTES]
    return f"<RULE>{body[:MAX_XML_BYTES]}</RULE>"


def main() -> None:
    fr = FederalRegisterClient(max_retries=4, sleep_between=0.5)
    reginfo = RegInfoClient()
    for (cid, cdate, source, agency, rins, docket, packet) in ROWS:
        out = HERE / cid
        ref = CommentRef(comment_id=cid, comment_date=cdate, source_document=source,
                         agency=agency, rins=rins, docket_id=docket,
                         packet_final_document=packet)
        write_json(out / "input.json", {
            "comment_id": cid, "comment_date": cdate, "source_document": source,
            "agency": agency, "rins": list(rins), "docket_id": docket,
            "packet_final_document": packet,
        })

        doc_numbers = set()
        if packet:
            details = fr.fetch_document_details(packet, enrich_identifiers=True)
            write_json(out / f"fr_doc_{slug(packet)}.json", details)
            doc_numbers.add(packet)
        for rin in rins:
            docs = fr.search_by_rin(rin)
            write_json(out / f"fr_rin_{slug(rin)}.json", docs)
            doc_numbers.update(d["document_number"] for d in docs or [])
        docs = fr.search_by_docket(docket)
        write_json(out / f"fr_docket_{slug(docket or '')}.json", docs)
        doc_numbers.update(d["document_number"] for d in docs or [])
        for identifier in fulltext_identifiers(ref):
            docs = fr.search_full_text(identifier)
            write_json(out / f"fr_term_{slug(identifier)}.json", docs)
            doc_numbers.update(d["document_number"] for d in docs or [])
        for rin in rins:
            write_json(out / f"agenda_{slug(rin)}.json", reginfo.fetch_unified_agenda(rin))
        for number in sorted(doc_numbers):
            xml = fr.fetch_document_full_text_xml(number)
            if not xml:
                continue
            path = out / f"xml_{slug(number)}.xml.gz"
            path.write_bytes(gzip.compress(trim_xml(xml).encode()))
        print(f"recorded {cid}: {len(doc_numbers)} documents")
    fr.close()
    reginfo.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Record the fixtures and write the fixture README**

Run: `python tests/fixtures/resolution/_record.py`
Expected: six `recorded <comment_id>: N documents` lines, no traceback.

Then create `tests/fixtures/resolution/README.md`:

```markdown
# Resolution acceptance fixtures

Recorded live on 2026-07-23 by `_record.py`. Each directory is one of the six
observed topologies; together they are the regression suite for the layer's
ontology of where a response can live.

- `input.json` — the CommentRef, taken from frozen snapshot `2026-07-15-ce44ac5`
  (`output/federal_register_2024_comments.csv`, `makeup/data/comments_raw_2024.csv`)
  and, for the DOT row, from `goldset/2026-07-17-ce44ac5/labeling_packet.csv`.
- `expected.json` — hand-written expectations, not recorded output.
- `fr_*.json` / `agenda_*.json` — verbatim API responses.
- `xml_*.xml.gz` — full-text XML, gzipped. Documents over 600 KB raw are trimmed
  to their `<SUPLINF>` container (only CMS `2025-14681`, 5.5 MB raw).

Re-record only when a fixture's expected behavior genuinely changes, and say so
in the commit message: these files are the reason the suite is deterministic.
`.csv` is Git-LFS-filtered in this repo — never add a `.csv` fixture here.
```

- [ ] **Step 3: Write the expectations and the failing acceptance test**

Create one `expected.json` per row, e.g. `tests/fixtures/resolution/DOT-OST-2024-0090-0049/expected.json`:

```json
{
  "status": "FOUND",
  "absence_reason": null,
  "qualifying_document_numbers": ["2024-29990"],
  "candidate_assertions": [
    {"document_number": "2025-02747", "relevance": "AGENCY_MISMATCH"},
    {"document_number": "2024-29990", "discovered_by": "RIN_SEARCH",
     "rule_class": "FINAL", "response_header_matched": false,
     "response_evidence": "STRONG"}
  ]
}
```

The other five, in full:

```json
// NOAA-NMFS-2023-0125-0016/expected.json
{
  "status": "FOUND",
  "absence_reason": null,
  "qualifying_document_numbers": ["2024-15931"],
  "candidate_assertions": [
    {"document_number": "2024-15931", "rule_class": "FINAL",
     "relevance": "MATCH", "response_evidence": "STRONG"}
  ]
}
```

```json
// BLM-2024-0001-0003/expected.json
{
  "status": "CONFIDENTLY_ABSENT",
  "absence_reason": "NO_VENUE_POSSIBLE",
  "qualifying_document_numbers": [],
  "candidate_assertions": [
    {"document_number": "2024-27333", "rule_class": "DIRECT_FINAL",
     "relevance": "MATCH"}
  ]
}
```

```json
// FBI-2024-0002-0006/expected.json
{
  "status": "CONFIDENTLY_ABSENT",
  "absence_reason": "RESPONSE_NOT_YET_PUBLISHED",
  "qualifying_document_numbers": [],
  "candidate_assertions": [
    {"document_number": "2024-28712", "rule_class": "PROPOSED"}
  ]
}
```

```json
// EPA-HQ-OAR-2022-0491-0022/expected.json
{
  "status": "CONFIDENTLY_ABSENT",
  "absence_reason": "NO_FINAL_RULE_PLANNED",
  "qualifying_document_numbers": [],
  "candidate_assertions": [
    {"document_number": "2024-04359", "rule_class": "PROPOSED"}
  ]
}
```

```json
// CMS-2024-0131-6043/expected.json
{
  "status": "FOUND",
  "absence_reason": null,
  "qualifying_document_numbers": ["2025-14681"],
  "candidate_assertions": [
    {"document_number": "2025-14681", "discovered_by": "FULLTEXT_SEARCH",
     "rule_class": "FINAL", "relevance": "MATCH"}
  ]
}
```

Create `tests/test_resolution_acceptance.py`:

```python
"""The six observed topologies, replayed offline from recorded fixtures."""

import gzip
import json
import re
from pathlib import Path

import pytest

from stratification_scripts.resolution.models import CommentRef, Status
from stratification_scripts.resolution.resolver import (
    DocumentResolver, qualifying_candidates,
)

FIXTURES = Path(__file__).parent / "fixtures" / "resolution"
ROWS = [
    "NOAA-NMFS-2023-0125-0016",
    "BLM-2024-0001-0003",
    "FBI-2024-0002-0006",
    "EPA-HQ-OAR-2022-0491-0022",
    "DOT-OST-2024-0090-0049",
    "CMS-2024-0131-6043",
]


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


class ReplayFRClient:
    """Serves recorded FR API responses. Raises on any unrecorded request."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory

    def _load(self, name: str):
        path = self.dir / name
        if not path.exists():
            raise AssertionError(f"unrecorded request: {name}")
        return json.loads(path.read_text())

    def fetch_document_details(self, document_number, enrich_identifiers=True):
        path = self.dir / f"fr_doc_{_slug(document_number)}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def fetch_document_full_text_xml(self, document_number):
        path = self.dir / f"xml_{_slug(document_number)}.xml.gz"
        return gzip.decompress(path.read_bytes()).decode() if path.exists() else None

    def search_by_rin(self, rin):
        return self._load(f"fr_rin_{_slug(rin)}.json")

    def search_by_docket(self, docket_id):
        return self._load(f"fr_docket_{_slug(docket_id or '')}.json")

    def search_full_text(self, identifier):
        return self._load(f"fr_term_{_slug(identifier)}.json")


class ReplayRegInfoClient:
    def __init__(self, directory: Path) -> None:
        self.dir = directory

    def fetch_unified_agenda(self, rin):
        path = self.dir / f"agenda_{_slug(rin)}.json"
        return json.loads(path.read_text()) if path.exists() else None


def _ref(directory: Path) -> CommentRef:
    data = json.loads((directory / "input.json").read_text())
    return CommentRef(
        comment_id=data["comment_id"], comment_date=data["comment_date"],
        source_document=data["source_document"], agency=data["agency"],
        rins=tuple(data["rins"]), docket_id=data["docket_id"],
        packet_final_document=data["packet_final_document"],
    )


@pytest.mark.parametrize("comment_id", ROWS)
def test_topology_fixture(comment_id):
    directory = FIXTURES / comment_id
    expected = json.loads((directory / "expected.json").read_text())
    resolver = DocumentResolver(
        fr_client=ReplayFRClient(directory),
        reginfo_client=ReplayRegInfoClient(directory),
    )
    result = resolver.resolve(_ref(directory))

    assert result.status.value == expected["status"], result.to_dict()
    actual_reason = result.absence_reason.value if result.absence_reason else None
    assert actual_reason == expected["absence_reason"], result.to_dict()
    assert [c.document_number for c in qualifying_candidates(result)] == \
        expected["qualifying_document_numbers"], result.to_dict()

    by_number = {c.document_number: c.to_dict() for c in result.candidates}
    for assertion in expected["candidate_assertions"]:
        number = assertion["document_number"]
        assert number in by_number, f"{number} missing from candidates"
        for key, value in assertion.items():
            if key == "document_number":
                continue
            assert by_number[number][key] == value, (number, key, by_number[number])


def test_dot_row_is_the_header_flag_regression():
    """A real response whose response header does not match must still be FOUND."""
    directory = FIXTURES / "DOT-OST-2024-0090-0049"
    resolver = DocumentResolver(
        fr_client=ReplayFRClient(directory),
        reginfo_client=ReplayRegInfoClient(directory),
    )
    result = resolver.resolve(_ref(directory))
    final = [c for c in result.candidates if c.document_number == "2024-29990"][0]
    assert final.response_header_matched is False
    assert result.status is Status.FOUND


def test_no_fixture_directory_is_missing():
    assert sorted(p.name for p in FIXTURES.iterdir() if p.is_dir()) == sorted(ROWS)
```

- [ ] **Step 4: Run the acceptance suite**

Run: `pytest tests/test_resolution_acceptance.py -v`
Expected: 8 passed.

If a row fails, fix the **layer**, not the expectation — each expectation is a hand-traced ground truth. The two most likely genuine surprises, and what to do:
- The BLM agenda shows a completed final action, so `NO_VENUE_POSSIBLE` depends only on the structural candidate and `withdrawn == False`. If the recorded agenda has `withdrawn: true`, verify on reginfo.gov before touching the corroboration rule.
- The EPA row needs `has_undetermined_final_rule` to be true in the recorded agenda. If it is false, the Task 1 fix did not take — re-run `pytest tests/test_reginfo_timetable.py` before changing anything here.

- [ ] **Step 5: Run the full suite**

Run: `pytest tests/ -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add tests/test_resolution_acceptance.py tests/fixtures/resolution/
git commit -m "test(resolution): six-topology offline acceptance suite + recorded fixtures"
```

---

### Task 13: CLI, snapshot inputs, and run outputs

**Files:**
- Create: `stratification_scripts/resolution/inputs.py`, `stratification_scripts/resolution/cli.py`
- Test: `tests/test_resolution_cli.py`

**Interfaces — Produces:**
- `inputs.refs_from_snapshot(snapshot_id: str, *, year: int = 2024, comment_ids: Optional[Sequence[str]] = None, limit: Optional[int] = None) -> List[CommentRef]`
- `inputs.refs_from_goldset_packet(seed_id: str) -> List[CommentRef]`
- `cli.write_run(results: List[ResolutionResult], manifest: dict, out_dir: Path) -> Path`
- `cli.main(argv=None) -> int` — `python -m stratification_scripts.resolution resolve --snapshot <id> [--year 2024] [--seed-id <id>] [--comment-id X ...] [--limit N] [--run-id <id>]`
- Outputs: `resolution/<run-id>/resolutions.jsonl`, `resolution/<run-id>/summary.csv`, `resolution/<run-id>/manifest.json`

**Why a `--seed-id` path:** the resolver is dual-use. Resolving exactly the goldset seed's rows is what turns it into an annotation-cost reduction — the packet can then show candidates, rule class, and evidence instead of making the labeler do RIN enumeration by hand.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_resolution_cli.py
import json

from stratification_scripts.resolution import cli
from stratification_scripts.resolution.models import (
    AgendaStatus, Channel, ResolutionResult, Status,
)


def _result(comment_id: str) -> ResolutionResult:
    return ResolutionResult(
        comment_id=comment_id, comment_date="2024-09-23", source_document="2024-18496",
        status=Status.FOUND, absence_reason=None, candidates=[],
        agenda=AgendaStatus("2105-AF05", "FINAL", [], False, False,
                            "2026-07-23T00:00:00", True),
        channels_run={c: "ok" for c in Channel}, resolved_at="2026-07-23T00:00:00",
    )


def test_write_run_emits_jsonl_summary_and_manifest(tmp_path):
    out = cli.write_run(
        results=[_result("A-1"), _result("B-2")],
        manifest={"snapshot": "2026-07-15-ce44ac5", "year": 2024, "rows": 2,
                  "cache_stats": {}, "started_at": "t0", "finished_at": "t1"},
        out_dir=tmp_path / "run1",
    )
    assert (out / "resolutions.jsonl").exists()
    assert (out / "summary.csv").exists()
    assert (out / "manifest.json").exists()
    lines = (out / "resolutions.jsonl").read_text().strip().splitlines()
    assert [json.loads(line)["comment_id"] for line in lines] == ["A-1", "B-2"]
    assert json.loads((out / "manifest.json").read_text())["rows"] == 2


def test_write_run_refuses_to_overwrite_a_non_empty_directory(tmp_path):
    target = tmp_path / "run1"
    target.mkdir()
    (target / "resolutions.jsonl").write_text("existing")
    try:
        cli.write_run(results=[_result("A-1")], manifest={}, out_dir=target)
    except FileExistsError:
        return
    raise AssertionError("expected FileExistsError")


def test_main_requires_a_snapshot():
    assert cli.main(["resolve"]) != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resolution_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: ...resolution.cli`.

- [ ] **Step 3: Implement**

```python
# stratification_scripts/resolution/inputs.py
"""Build CommentRefs from a frozen snapshot (or a goldset seed packet)."""

from __future__ import annotations

from typing import List, Optional, Sequence

import polars as pl

from .. import config
from .models import CommentRef


def _rins(row: dict) -> tuple:
    raw = str(row.get("rin_all") or row.get("rin") or "")
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    return tuple(dict.fromkeys(p for p in parts if p and p.lower() not in ("none", "null")))


def _clean(value) -> Optional[str]:
    text = str(value or "").strip()
    return None if text.lower() in ("", "none", "null") else text


def refs_from_snapshot(
    snapshot_id: str,
    *,
    year: int = 2024,
    comment_ids: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> List[CommentRef]:
    """One CommentRef per comment, joined from the snapshot's comment + FR tables."""
    base = config.get_frozen_snapshot_path(snapshot_id)
    comments = pl.read_csv(
        base / f"makeup/data/comments_raw_{year}.csv", infer_schema_length=0
    ).select(["comment_id", "document_number", "posted_date", "receive_date"])
    fr = pl.read_csv(
        base / f"output/federal_register_{year}_comments.csv", infer_schema_length=0
    ).select([
        "document_number", "agency", "docket_id", "rin", "rin_all",
        "final_rule_document_number",
    ])
    if comment_ids:
        comments = comments.filter(pl.col("comment_id").is_in(list(comment_ids)))
    joined = comments.join(fr, on="document_number", how="left")
    if limit:
        joined = joined.head(limit)

    refs: List[CommentRef] = []
    for row in joined.iter_rows(named=True):
        posted = _clean(row.get("posted_date")) or _clean(row.get("receive_date")) or ""
        refs.append(CommentRef(
            comment_id=str(row["comment_id"]),
            comment_date=posted[:10],
            source_document=str(row.get("document_number") or ""),
            agency=str(row.get("agency") or ""),
            rins=_rins(row),
            docket_id=_clean(row.get("docket_id")),
            packet_final_document=_clean(row.get("final_rule_document_number")),
        ))
    return refs


def refs_from_goldset_packet(seed_id: str) -> List[CommentRef]:
    """CommentRefs for exactly the rows in a goldset seed's labeling packet.

    The dual-use path: resolving the seed is what lets the annotation packet ship
    candidates instead of making the labeler enumerate RINs by hand.
    """
    packet_path = config.get_goldset_seed_path(seed_id) / "labeling_packet.csv"
    packet = pl.read_csv(packet_path, infer_schema_length=0)
    refs: List[CommentRef] = []
    for row in packet.iter_rows(named=True):
        refs.append(CommentRef(
            comment_id=str(row["comment_id"]),
            comment_date="",            # filled by the caller from the snapshot
            source_document=str(row.get("document_number") or ""),
            agency=str(row.get("agency") or ""),
            rins=_rins(row),
            docket_id=_clean(row.get("docket_id")),
            packet_final_document=_clean(row.get("final_rule_document_number")),
        ))
    return refs
```

```python
# stratification_scripts/resolution/cli.py
"""`python -m stratification_scripts.resolution resolve` — standalone, like goldset."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import polars as pl

from .. import config
from ..federal_register.client import FederalRegisterClient
from ..logging_utils import get_logger
from ..reginfo.client import RegInfoClient
from .inputs import refs_from_goldset_packet, refs_from_snapshot
from .models import CommentRef, ResolutionResult
from .resolver import DocumentResolver, qualifying_candidates

logger = get_logger(__name__)


def write_run(*, results: List[ResolutionResult], manifest: dict, out_dir: Path) -> Path:
    """Write resolutions.jsonl + summary.csv + manifest.json. Never overwrites."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "resolutions.jsonl").open("w") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict()) + "\n")

    rows = []
    for result in results:
        quals = qualifying_candidates(result)
        rows.append({
            "comment_id": result.comment_id,
            "status": result.status.value,
            "absence_reason": result.absence_reason.value if result.absence_reason else "",
            "n_candidates": len(result.candidates),
            "n_qualifying": len(quals),
            "first_qualifying": quals[0].document_number if quals else "",
            "channels_failed": ";".join(
                c.value for c, state in result.channels_run.items() if state != "ok"
            ),
        })
    pl.DataFrame(rows).write_csv(out_dir / "summary.csv")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return out_dir


def _resolve_refs(refs: List[CommentRef]) -> tuple[List[ResolutionResult], dict]:
    fr = FederalRegisterClient(max_retries=6, sleep_between=0.4)
    reginfo = RegInfoClient()
    resolver = DocumentResolver(fr_client=fr, reginfo_client=reginfo)
    results: List[ResolutionResult] = []
    try:
        for index, ref in enumerate(refs, start=1):
            results.append(resolver.resolve(ref))
            if index % 10 == 0:
                logger.info(f"resolved {index}/{len(refs)}")
    finally:
        fr.close()
        reginfo.close()
    return results, resolver.cache.stats


def cmd_resolve(args) -> int:
    if args.seed_id:
        seed_refs = refs_from_goldset_packet(args.seed_id)
        dated = {
            r.comment_id: r for r in refs_from_snapshot(
                args.snapshot, year=args.year,
                comment_ids=[r.comment_id for r in seed_refs],
            )
        }
        refs = [dated.get(r.comment_id, r) for r in seed_refs]
    else:
        refs = refs_from_snapshot(
            args.snapshot, year=args.year,
            comment_ids=args.comment_id or None, limit=args.limit,
        )
    if not refs:
        logger.error("no comments selected")
        return 2

    started_at = datetime.now().isoformat()
    results, cache_stats = _resolve_refs(refs)
    run_id = args.run_id or f"{datetime.now():%Y-%m-%d}-{args.snapshot}"
    out_dir = write_run(
        results=results,
        manifest={
            "snapshot": args.snapshot, "year": args.year, "seed_id": args.seed_id,
            "rows": len(results), "cache_stats": cache_stats,
            "started_at": started_at, "finished_at": datetime.now().isoformat(),
            "note": "agenda data is time-varying; re-runs can differ",
        },
        out_dir=config.get_resolution_run_path(run_id),
    )
    logger.info(f"wrote {len(results)} resolutions to {out_dir}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m stratification_scripts.resolution")
    sub = parser.add_subparsers(dest="command", required=True)

    p_resolve = sub.add_parser("resolve", help="Resolve response venues for comments.")
    p_resolve.add_argument("--snapshot", required=True, help="Frozen snapshot id.")
    p_resolve.add_argument("--year", type=int, default=2024)
    p_resolve.add_argument("--seed-id", default=None,
                           help="Resolve exactly a goldset seed's packet rows.")
    p_resolve.add_argument("--comment-id", action="append", default=[],
                           help="Repeatable; resolve specific comment ids.")
    p_resolve.add_argument("--limit", type=int, default=None)
    p_resolve.add_argument("--run-id", default=None)
    p_resolve.set_defaults(func=cmd_resolve)

    args = parser.parse_args(argv)
    return args.func(args)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_resolution_cli.py -v`
Expected: 3 passed.

- [ ] **Step 5: Live smoke run on the six fixture rows**

Run:
```bash
python -m stratification_scripts.resolution resolve \
  --snapshot 2026-07-15-ce44ac5 \
  --comment-id NOAA-NMFS-2023-0125-0016 \
  --comment-id BLM-2024-0001-0003 \
  --comment-id FBI-2024-0002-0006 \
  --comment-id EPA-HQ-OAR-2022-0491-0022 \
  --comment-id CMS-2024-0131-6043 \
  --run-id smoke-six
```
Expected: `wrote 5 resolutions to .../resolution/smoke-six`. (DOT is absent from `comments_raw_2024` — it comes only from the goldset packet, which is why the `--seed-id` path exists. Do not add it to this command.)

Then inspect and **report the actual status/absence_reason for each row in the task summary**, alongside the live-vs-fixture difference for the EPA row (its agenda is time-varying by construction):

```bash
cat resolution/smoke-six/summary.csv
```

- [ ] **Step 6: Run the full suite**

Run: `pytest tests/ -q`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add stratification_scripts/resolution/inputs.py stratification_scripts/resolution/cli.py tests/test_resolution_cli.py
git commit -m "feat(resolution): resolve CLI over frozen snapshots + goldset seeds"
```

---

## Not in this plan (deliberately)

Each is a separate spec/plan; none blocks this one.

- **Wiring into `track_responses`.** Replacing the faith-based `final_doc_number` lookup and emitting typed outcomes instead of a silent degrade to web-search is a separately-measured change. `qualifying_candidates()` is the contract it will consume.
- **The gold-set frame swap and redraw.** The corrected predicate is `status == FOUND`, but changing frame membership invalidates the 378-row frame, the 30-row draw, and the HT weights (`response_sample_weight`, `frame_weight_mass`, `projected_missed`) — a redraw is required. **The labels survive** as per-comment ground truth; they simply stop being a probability sample. This is *not* a reason to pause the Arvind handoff — only the estimator needs rebuilding.
- **Annotation packet v2.** Emitting all candidates with `rule_class`, date, `response_evidence`, and `discovered_by`, with the human-superset constraint. Per the spec, packet v2 is where the **goldset-harness walkthrough debt gates**, since it stacks on unwalked packet design decisions.
- **Retiring or caveating `lifecycle_stage`,** and its `AUDIT_FINDINGS` entry for the reproducibility hazard (same comments → different values on re-run).
- **Marginal-yield thresholds / the empirical stopping rule.** Measurable only once gold-set labels exist.

## Decision Ledger (living — canonical location for this build)

- **L1. Standalone package, dependency-injected clients.** `DocumentResolver(fr_client=…, reginfo_client=…)` rather than constructing clients internally. Buys the offline six-topology suite, which is the acceptance criterion. Rejected module-level client singletons (untestable without network). Reopen if a caller needs process-wide connection pooling.
- **L2. `date_raw` added alongside `date` rather than storing a raw string in `date`.** The spec floated putting `"To Be Determined"` in `date`; that would inflate `timetable_action_count` (which counts non-empty `date`) and pollute the chronological sort in every existing output column. Additive key = zero blast radius. Reopen if a consumer needs the raw value in the primary field.
- **L3. Scoped row-wise timetable parse, with the legacy global regex as fallback.** Widening the old two-cell regex to accept non-dates would have matched arbitrary `<td>label</td><td>value</td>` pairs across the whole reginfo page. Scoping to the `Timetable:` table also fixes citation extraction (same-row third cell instead of "the next `<td>` in the document"). Fallback preserves behavior on pages whose markup differs. Reopen if reginfo changes the table markup.
- **L4. Lineage check applies to `PACKET_LINK` only; the agency check applies to every channel.** Channels 2/3/5 query *by* an identifier from the comment's own lineage, so a hit is itself lineage evidence. Applying a global lineage rule would reject CMS `2025-14681` (different RIN, different docket) — deleting the single topology this layer exists to recover. Rejected: global lineage (kills CMS), no lineage at all (lets the FCC link through on agency-only checks for same-agency mistakes). Reopen if a wide channel starts returning same-agency noise.
- **L5. `FOUND` on `WEAK` evidence, `UNKNOWN` only on `NONE`.** *The spec is internally inconsistent here:* its `FOUND` clause says "≥1 qualifying candidate with `response_evidence != NONE`" while its `UNKNOWN` clause says a qualifying candidate with `NONE`/`WEAK` → `UNKNOWN`. Resolved toward the `FOUND` clause: `FOUND` is not an assertion that a response exists, only that a venue with readable text was resolved and can be handed to judgment. Routing `WEAK` to `UNKNOWN` would hide a real venue from the judgment step that exists to read it. `response_evidence` is on every candidate, so any consumer wanting the stricter rule can filter. **Flagged to Jonathan — the spec needs a one-line amendment either way.**
- **L6. `NO_VENUE_POSSIBLE` additionally requires the structural candidate's own evidence to be non-`STRONG`.** *A refinement beyond the spec's corroboration table.* A direct final rule that actually carries a comments-and-responses discussion cannot support "no venue possible." Costs at most one XML fetch per structural candidate (hence `DIRECT_FINAL` and `CONFIRMATION_OF_EFFECTIVE_DATE` are in `FETCHABLE_CLASSES`). Does not affect the BLM fixture, whose DFR predates the comment and is therefore never fetched. **Flagged to Jonathan — veto returns the rule to pure structure.**
- **L7. Absence-reason precedence `NO_VENUE_POSSIBLE` → `NO_FINAL_RULE_PLANNED` → `RESPONSE_NOT_YET_PUBLISHED`.** Forced by the fixtures: EPA has both an NPRM and a TBD agenda and must land on `NO_FINAL_RULE_PLANNED`; FBI has an NPRM with a normally-scheduled agenda and must land on `RESPONSE_NOT_YET_PUBLISHED`. Reopen if a fixture arises where a structural candidate coexists with a genuine "not yet."
- **L8. Merge keeps the most precise channel, then re-derives relevance.** Without re-derivation, a document found by both `FULLTEXT_SEARCH` and `PACKET_LINK` would keep the packet-link classification and could be falsely rescued (or falsely rejected). Reopen if per-channel provenance for *every* discovering channel becomes necessary — the model currently records one.
- **L9. `None` vs `[]` from the search methods is the channel-state signal.** `[]` is a clean run (`"ok"`), `None` is a failure (`"failed:…"`). Zero-result docket queries are common and legitimate (NOAA files under a docket string the FR API does not index) and must never block an absence claim on their own. Reopen never — collapsing these is the silent-failure mode in miniature.
- **L10. Oversized fixture XML is trimmed to its `<SUPLINF>` container.** CMS `2025-14681` is 5.5 MB raw / 1.1 MB gzipped; `extract_response_section` only reads the `<SUPLINF>` container, so a trimmed fixture exercises the same code path. Documented in `tests/fixtures/resolution/README.md` with the 600 KB threshold. Rejected committing the full XML (repo bloat) and skipping the CMS fixture (it is the cross-RIN topology). Reopen if a test needs structure outside `SUPLINF`.

## Self-Review

**1. Spec coverage.** Problem defects 1–3 → Tasks 1 (defect 3), 6 + "Not in this plan" (defects 1–2, frame swap deferred by design). Goal/acceptance → Task 12 (all six rows, including CMS cross-RIN recovery and EPA confident absence). Non-goals → Global Constraints (no judgment, no cli wiring, no Ruler B, no LLM-path rewrite). Five channels → Tasks 9 (1–3), 10 (4), 11 (5). Channel-5 identifier constraint → Task 11 + `search_full_text`. Data contract → Task 2 (every field present, incl. `channels_run`, `agenda`, `response_section_ref`). Response-evidence grading → Task 5. Status semantics F1–F4 → Task 6. `absence_reason` + per-reason corroboration → Task 6. Rule classification from `action` → Task 3. Chronology rule → Task 4. Relevance check → Task 4. Consumers → `qualifying_candidates()` (Task 10) + the explicit deferral list. Fetch policy + cross-row cache → Tasks 8, 10. Reuse-vs-build split → honored (reginfo/extractor/FR client reused; searches, filters, classification, composition built). Prerequisite fix incl. dedup key and citation path → Task 1. Two-tier testing → unit Tasks 3–9, fixture/golden Task 12; no live-network test is required for CI (the only live step is Task 13 Step 5, a manual smoke run). Open questions → all four listed under "Not in this plan."

**2. Placeholder scan.** No TBD/TODO, no "add error handling", no "similar to Task N", no test-free steps. Two steps intentionally depend on recorded reality rather than literal pre-written content: Task 12 Step 2 (fixture bytes come from the recorder, whose full source is given) and Task 5's threshold-tuning note (bounded by two committed fixture strings and a required ledger entry). Both pin the interface, the test, and the decision rule.

**3. Type consistency.** `CommentRef`/`CandidateDocument`/`AgendaStatus`/`ResolutionResult` field names identical across Tasks 2, 9, 10, 12, 13. `ChannelOutcome(candidates, state)` consistent across Tasks 9, 10, 11. `rule_class_from_action(action, doc_type)` called with both arguments in Task 9 exactly as defined in Task 3. `relevance_of(*, discovered_by, agency_names, rins, docket_id, ref)` keyword-only in Tasks 4, 9, 10. `response_evidence_from_extract(extract)` used in Task 10 as defined in Task 5. `derive_status(candidates, agenda, channels_run)` and `qualifies(candidate)` identical in Tasks 6, 10. `search_by_rin`/`search_by_docket`/`search_full_text`/`fetch_document_details`/`fetch_document_full_text_xml` are the complete client surface, implemented in Task 7 and re-implemented identically by `ReplayFRClient` in Task 12. `CHANNEL_OK` (Task 6) and `STATE_OK` (Task 9) are the same string `"ok"`, deliberately named per-module.
