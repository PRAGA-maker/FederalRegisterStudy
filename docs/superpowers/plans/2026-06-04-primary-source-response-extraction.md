# Primary-Source Response Extraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ground Step-5 agency-response classification on the primary-source Final-Rule preamble "Response to Comments" section (and optional docket RTC docs), using Grok without web search when a linked Final Rule exists; keep web search as fallback. Output stays strictly **per-comment**.

**Architecture:** A pure parser (`extract_response_section`) isolates the comment-response discussion from FR `full_text_xml`; the FR client gains an XML fetcher; the xAI tracker gains a no-search grounded classify method; `track_responses` builds a per-rule grounded-text cache and routes each sampled comment to grounded-vs-web-search, writing new provenance columns. Validated: 19/19 present-response rules captured on a 37-rule heterogeneity sample (`docs/superpowers/specs/2026-06-04-primary-source-response-extraction-design.md`).

**Tech Stack:** Python 3.10+, polars, requests, pydantic, openai SDK (xAI base URL), pytest + pytest-asyncio. Run tests with `python -m pytest` (install dev deps first: `python -m pip install pytest pytest-asyncio` or `uv sync --extra dev`).

**Reference:** the validated algorithm lives in `prd/govinfo_probe/extract_proto.py` — port it, don't reinvent.

---

## File structure

- Create `stratification_scripts/makeup/fr_response_extractor.py` — `ResponseExtract` dataclass, `extract_response_section(xml)`, `find_docket_rtc_documents(...)`. One responsibility: turn raw FR XML / a docket into grounded evidence text.
- Modify `stratification_scripts/federal_register/client.py` — add `fetch_document_full_text_xml(document_number)`.
- Modify `stratification_scripts/gemini_client.py` — add `GROUNDED_RESPONSE_PROMPT`.
- Modify `stratification_scripts/xai_response_client.py` — add `track_response_grounded(...)` + `track_grounded_batch(...)`.
- Modify `stratification_scripts/makeup/track_responses.py` — cache build + grounded/fallback routing + new columns.
- Modify `stratification_scripts/config.py` — add `enable_primary_source_grounding: bool = True`, `grounded_max_chars: int = 100_000`.
- Create `tests/__init__.py`, `tests/conftest.py`, `tests/test_fr_response_extractor.py`, `tests/test_xai_grounded.py`, `tests/test_fr_client_xml.py`.

---

## Task 1: `extract_response_section` core parser

**Files:**
- Create: `stratification_scripts/makeup/fr_response_extractor.py`
- Create: `tests/__init__.py`, `tests/conftest.py`, `tests/test_fr_response_extractor.py`

- [ ] **Step 1: Create test scaffolding + failing tests**

Create `tests/__init__.py` (empty). Create `tests/conftest.py`:

```python
import sys
from pathlib import Path

# Ensure the package is importable when running pytest from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

Create `tests/test_fr_response_extractor.py`:

```python
from stratification_scripts.makeup.fr_response_extractor import extract_response_section, GROUND_CAP

# A small final-rule XML with an explicit, substantial response section.
XML_RESP_HD = """<RULE><PREAMB>
<HD SOURCE="HED">SUPPLEMENTARY INFORMATION:</HD>
<SUPLINF>
<HD SOURCE="HD1">I. Background</HD>
<P>The agency proposed X.</P>
<HD SOURCE="HD1">III. Response to Comments</HD>
<P>We received 500 comments. We agree with commenters that X should change. """ + ("blah " * 600) + """</P>
<HD SOURCE="HD1">IV. Regulatory Analysis</HD>
<P>Executive Order 12866 review.</P>
</SUPLINF></PREAMB></RULE>"""

# Comment-by-topic: response head followed by same-level "Comments on X" heads.
XML_BY_TOPIC = """<SUPLINF>
<HD SOURCE="HD1">Response to Comments</HD>
<P>intro.</P>
<HD SOURCE="HD1">Comments on Scope</HD><P>""" + ("a " * 400) + """We adopt this.</P>
<HD SOURCE="HD1">Comments on Cost</HD><P>""" + ("b " * 400) + """We decline.</P>
<HD SOURCE="HD1">Regulatory Flexibility Act</HD><P>not a comment.</P>
</SUPLINF>"""

# Tiny pointer head -> should NOT be used as the section (falls to whole preamble).
XML_POINTER = """<SUPLINF>
<HD SOURCE="HD1">Public Comments Received</HD><P>We received 3,500 comments; responses appear under the relevant headings below.</P>
<HD SOURCE="HD1">Payment Rates</HD><P>""" + ("c " * 500) + """We agree with the commenter on rates.</P>
</SUPLINF>"""

# No SUPLINF, but PREAMB present (DOE/FHWA style).
XML_PREAMB_ONLY = """<PREAMB>
<HD SOURCE="HED">AGENCY:</HD><P>DOE.</P>
<HD SOURCE="HED">SUPPLEMENTARY INFORMATION:</HD>
<HD SOURCE="HD1">Discussion of Comments</HD><P>""" + ("d " * 600) + """We disagree with the commenter.</P>
</PREAMB>"""

# No comment discussion at all (technical correction).
XML_NO_RESPONSE = """<SUPLINF><HD SOURCE="HD1">Need for Correction</HD><P>This corrects a typo.</P></SUPLINF>"""

def test_explicit_response_section_is_isolated():
    r = extract_response_section(XML_RESP_HD)
    assert r.found_response_hd is True
    assert r.matched_header.startswith("III. Response to Comments")
    assert "We agree with commenters" in r.grounded_text
    assert "Executive Order 12866" not in r.grounded_text  # stops at next section
    assert r.method == "response_hd"

def test_comment_by_topic_block_is_kept_whole():
    r = extract_response_section(XML_BY_TOPIC)
    assert "We adopt this." in r.grounded_text
    assert "We decline." in r.grounded_text          # same-level comment heads kept
    assert "Regulatory Flexibility Act" not in r.grounded_text  # non-comment head ends it

def test_pointer_head_falls_through_to_whole_preamble():
    r = extract_response_section(XML_POINTER)
    # tiny pointer section (< POINTER_MIN) must not be the grounded_text alone
    assert "We agree with the commenter on rates" in r.grounded_text
    assert r.method in ("suplinf_full", "response_hd_to_full")

def test_preamb_fallback_when_no_suplinf():
    r = extract_response_section(XML_PREAMB_ONLY)
    assert r.suplinf_len > 0
    assert "We disagree with the commenter" in r.grounded_text

def test_no_response_returns_preamble_but_short():
    r = extract_response_section(XML_NO_RESPONSE)
    assert r.found_response_hd is False
    assert "corrects a typo" in r.grounded_text

def test_grounded_text_never_exceeds_cap():
    big = "<SUPLINF>" + "".join(
        f'<HD SOURCE="HD1">Comments on Topic {i}</HD><P>{"word " * 2000} we agree.</P>' for i in range(80)
    ) + "</SUPLINF>"
    r = extract_response_section(big)
    assert r.grounded_len <= GROUND_CAP

def test_missing_xml_is_safe():
    assert extract_response_section("").grounded_text == ""
    assert extract_response_section("<RULE><REGTEXT>no preamble</REGTEXT></RULE>").method == "no_preamble"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fr_response_extractor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stratification_scripts.makeup.fr_response_extractor'`.

- [ ] **Step 3: Implement `fr_response_extractor.py` (port from `prd/govinfo_probe/extract_proto.py`)**

Create `stratification_scripts/makeup/fr_response_extractor.py`:

```python
"""Extract the primary-source agency 'Response to Comments' discussion from a
Federal Register Final Rule's structured full_text_xml, to feed an LLM that
classifies per-comment agency dispositions. Validated in prd/govinfo_probe/.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from stratification_scripts.logging_utils import get_logger

logger = get_logger(__name__)

GROUND_CAP = 100_000   # max chars of grounded evidence fed to the LLM
POINTER_MIN = 2_500    # a matched response head shorter than this may be just a pointer

RESP_HD = re.compile(
    r"(responses?\s+to\s+(the\s+)?(public\s+|significant\s+|major\s+)?comments?"
    r"|comments?\s+and\s+responses?"
    r"|summary\s+(and\s+analysis\s+)?of\s+(the\s+)?(public\s+)?comments?(\s+and\s+responses?)?"
    r"|(public\s+)?comments?\s+(received|and\s+(agency\s+)?responses?|and\s+summary)"
    r"|discussion\s+(and\s+(analysis|responses?)\s+)?of\s+(the\s+)?(public\s+)?comments?"
    r"|(public\s+)?comments?\s+and\s+(the\s+)?(department|agency|secretary)"
    r"|analysis\s+(and\s+responses?\s+)?of\s+(public\s+)?comments?"
    r"|response\s+to\s+(the\s+)?(public\s+)?comment"
    r"|comments?\s+(on|received\s+on)\s+the\s+(proposed|interim|final|nprm)"
    r"|public\s+comments?\s+and\b"
    r"|agency\s+responses?\s+to\s+comments?"
    r"|comments?\s+and\s+(the\s+)?(department|agency)'?s?\s+responses?)", re.I)
COMMENTISH = re.compile(r"\bcomment", re.I)
DENSITY_KW = re.compile(
    r"\b(comment(s|er|ers)?|in\s+response|we\s+(agree|disagree|adopt(ed)?|decline[d]?|considered"
    r"|are\s+(not\s+)?(adopting|persuaded))|the\s+(agency|department|commission|commenter))\b", re.I)
BOILER = re.compile(
    r"^\s*(AGENCY|ACTION|SUMMARY|DATES|ADDRESSES|FOR FURTHER|SUPPLEMENTARY|EFFECTIVE|TABLE OF CONTENTS)\b", re.I)


@dataclass
class ResponseExtract:
    grounded_text: str
    method: str                       # response_hd | response_hd_to_full | suplinf_full | comment_density | no_preamble
    matched_header: Optional[str]
    found_response_hd: bool
    suplinf_len: int
    grounded_len: int


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()


def _hd_level(tag_open: str) -> int:
    m = re.search(r'SOURCE="HD(\d+)"', tag_open, re.I)
    if m:
        return int(m.group(1))
    if re.search(r'SOURCE="HED"', tag_open, re.I):
        return 0
    return 99


def _select_by_comment_density(body: str, hds, cap: int) -> str:
    chunks = []
    for i, m in enumerate(hds):
        start = m.start()
        end = hds[i + 1].start() if i + 1 < len(hds) else len(body)
        txt = _clean(body[start:end])
        if txt:
            chunks.append((start, txt, len(DENSITY_KW.findall(txt))))
    cand = sorted([c for c in chunks if c[2] > 0], key=lambda c: c[2], reverse=True)
    picked, total = [], 0
    for start, txt, _hits in cand:
        if picked and total + len(txt) > cap:
            continue
        picked.append((start, txt)); total += len(txt)
        if total >= cap:
            break
    picked.sort(key=lambda c: c[0])
    return ("\n\n[...]\n\n".join(t for _, t in picked))[:cap]


def extract_response_section(xml: str) -> ResponseExtract:
    if not xml:
        return ResponseExtract("", "no_preamble", None, False, 0, 0)
    sup = re.search(r"<SUPLINF\b[^>]*>(.*?)</SUPLINF>", xml, re.S | re.I)
    if sup:
        body = sup.group(1)
    else:
        pre = re.search(r"<PREAMB\b[^>]*>(.*?)</PREAMB>", xml, re.S | re.I)
        if not pre:
            return ResponseExtract("", "no_preamble", None, False, 0, 0)
        pbody = pre.group(1)
        si = re.search(r"<HD\b[^>]*>\s*SUPPLEMENTARY INFORMATION", pbody, re.I)
        body = pbody[si.start():] if si else pbody
    suplinf_len = len(_clean(body))
    hds = list(re.finditer(r"<HD\b([^>]*)>(.*?)</HD>", body, re.S | re.I))

    section: Optional[str] = None
    matched_header: Optional[str] = None
    found = False
    for i, m in enumerate(hds):
        lvl = _hd_level(m.group(1)); txt = _clean(m.group(2))
        if not txt or BOILER.match(txt):
            continue
        if RESP_HD.search(txt):
            found = True; matched_header = txt
            start = m.start(); end = len(body)
            for j in range(i + 1, len(hds)):
                if _hd_level(hds[j].group(1)) <= lvl and not COMMENTISH.search(_clean(hds[j].group(2))):
                    end = hds[j].start(); break
            section = _clean(body[start:end])
            break

    full_preamble = _clean(body)
    if section is not None and POINTER_MIN <= len(section) <= GROUND_CAP:
        method, grounded = "response_hd", section
    elif len(full_preamble) <= GROUND_CAP:
        method = "response_hd_to_full" if section is not None else "suplinf_full"
        grounded = full_preamble
    else:
        method, grounded = "comment_density", _select_by_comment_density(body, hds, GROUND_CAP)
    grounded = grounded[:GROUND_CAP]
    return ResponseExtract(grounded, method, matched_header, found, suplinf_len, len(grounded))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fr_response_extractor.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/makeup/fr_response_extractor.py tests/
git commit -m "feat(responses): add FR preamble response-section extractor (validated parser)"
```

---

## Task 2: RTC-document harvester

**Files:**
- Modify: `stratification_scripts/makeup/fr_response_extractor.py`
- Modify: `tests/test_fr_response_extractor.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fr_response_extractor.py`:

```python
from stratification_scripts.makeup.fr_response_extractor import is_rtc_title

def test_rtc_title_matching():
    assert is_rtc_title("Response to Comments (22-4.5e)")
    assert is_rtc_title("Summary of Public Comments and Responses")
    assert is_rtc_title("RTC Document")
    assert not is_rtc_title("Regulatory Impact Analysis")
    assert not is_rtc_title("Economic Analysis")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_fr_response_extractor.py::test_rtc_title_matching -v`
Expected: FAIL — `ImportError: cannot import name 'is_rtc_title'`.

- [ ] **Step 3: Implement RTC helpers**

Append to `stratification_scripts/makeup/fr_response_extractor.py`:

```python
RTC_TITLE = re.compile(r"(response\s+to\s+comment|comment.{0,12}response|\bRTC\b|summary\s+of\s+(public\s+)?comment)", re.I)
RTC_MAX_CHARS = 60_000


def is_rtc_title(title: str) -> bool:
    return bool(title and RTC_TITLE.search(title))


def find_docket_rtc_documents(docket_id: str, regs_client, max_chars: int = RTC_MAX_CHARS) -> List[dict]:
    """Return [{document_id, title, text}] for docket 'Supporting & Related Material'
    documents whose titles look like standalone Response-to-Comments docs. Sparse
    (~1/8 EPA dockets); best-effort, never raises."""
    out: List[dict] = []
    if not docket_id or regs_client is None:
        return out
    try:
        resp = regs_client.request_json(
            "https://api.regulations.gov/v4/documents",
            params={"filter[docketId]": docket_id,
                    "filter[documentType]": "Supporting & Related Material",
                    "page[size]": 100},
        )
    except Exception as e:  # noqa: BLE001 — enrichment is best-effort
        logger.debug(f"RTC lookup failed for {docket_id}: {e}")
        return out
    for d in (resp or {}).get("data", []):
        attrs = d.get("attributes", {}) or {}
        title = attrs.get("title", "") or ""
        if not is_rtc_title(title):
            continue
        out.append({"document_id": d.get("id"), "title": title, "text": ""})
    return out
```

> Note: `request_json` is the existing `RegsGovClient` method (see `regulations_gov/client.py:295`). Downloading the RTC body text reuses `regs_client.get_document_detail` + attachment download in Task 5; this task only finds candidates by title (the testable, deterministic part).

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_fr_response_extractor.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/makeup/fr_response_extractor.py tests/test_fr_response_extractor.py
git commit -m "feat(responses): add docket RTC-document title finder"
```

---

## Task 3: `FederalRegisterClient.fetch_document_full_text_xml`

**Files:**
- Modify: `stratification_scripts/federal_register/client.py` (add method after `fetch_document_full_text`, ~line 468)
- Create: `tests/test_fr_client_xml.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fr_client_xml.py`:

```python
from unittest.mock import MagicMock
from stratification_scripts.federal_register.client import FederalRegisterClient

def test_fetch_full_text_xml_uses_details_url(monkeypatch):
    c = FederalRegisterClient(max_retries=1, sleep_between=0)
    monkeypatch.setattr(c, "fetch_document_details",
                        lambda dn, enrich_identifiers=False: {"full_text_xml_url": "http://x/doc.xml"})
    resp = MagicMock(status_code=200, text="<RULE><SUPLINF>hi</SUPLINF></RULE>")
    monkeypatch.setattr(c.session, "get", lambda url, timeout=60: resp)
    assert "<SUPLINF>" in c.fetch_document_full_text_xml("2024-19696")

def test_fetch_full_text_xml_missing_url_returns_none(monkeypatch):
    c = FederalRegisterClient(max_retries=1, sleep_between=0)
    monkeypatch.setattr(c, "fetch_document_details", lambda dn, enrich_identifiers=False: {})
    assert c.fetch_document_full_text_xml("2024-19696") is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_fr_client_xml.py -v`
Expected: FAIL — `AttributeError: 'FederalRegisterClient' object has no attribute 'fetch_document_full_text_xml'`.

- [ ] **Step 3: Implement the method**

In `stratification_scripts/federal_register/client.py`, add this method to `FederalRegisterClient` immediately after `fetch_document_full_text` (after line 468):

```python
    def fetch_document_full_text_xml(self, document_number: str) -> Optional[str]:
        """Fetch the STRUCTURED full-text XML (<SUPLINF>/<HD> markup) for an FR
        document. Returns raw XML string, or None on any failure."""
        if not document_number:
            return None
        details = self.fetch_document_details(document_number, enrich_identifiers=False)
        if not details:
            return None
        xml_url = details.get("full_text_xml_url")
        if not xml_url:
            logger.warning(f"No full_text_xml_url for document {document_number}")
            return None
        if self.sleep_between > 0:
            time.sleep(self.sleep_between)
        backoff = 1.0
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(xml_url, timeout=60)
            except requests.RequestException:
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(backoff); backoff = min(backoff * 2, 16); continue
            if r.status_code == 200:
                return r.text or None
            if r.status_code in (403, 429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                time.sleep(backoff); backoff = min(backoff * 2, 16); continue
            return None
        return None
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_fr_client_xml.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/federal_register/client.py tests/test_fr_client_xml.py
git commit -m "feat(fr): add fetch_document_full_text_xml for structured preamble"
```

---

## Task 4: Grounded prompt + grounded xAI classify method

**Files:**
- Modify: `stratification_scripts/gemini_client.py` (add `GROUNDED_RESPONSE_PROMPT` after `RESPONSE_TRACKING_PROMPT`, ~line 175)
- Modify: `stratification_scripts/xai_response_client.py`
- Create: `tests/test_xai_grounded.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_xai_grounded.py`:

```python
import asyncio
from stratification_scripts.gemini_client import GROUNDED_RESPONSE_PROMPT, AgencyResponse
from stratification_scripts.xai_response_client import XAIResponseTracker

def test_grounded_prompt_includes_evidence_and_comment():
    p = GROUNDED_RESPONSE_PROMPT.format(
        comment_id="C1", document_number="2024-1", agency="EPA",
        commenter_type="industry", submission_date="2024-01-01",
        full_comment_text="please change rule X",
        grounded_text="We agree with commenters on X and adopt the change.")
    assert "please change rule X" in p
    assert "We agree with commenters on X" in p
    assert "this comment" in p.lower()

def test_grounded_tracker_disables_search_and_returns_schema(monkeypatch):
    tracker = XAIResponseTracker(api_key="x", enable_search=True)  # search on for fallback

    class FakeParsed:
        output_parsed = AgencyResponse(response_found="yes", agency_decision="accept",
                                       response_text="We agree.", response_location="N/A",
                                       reasoning="grounded")
        output_text = "{}"
    captured = {}
    def fake_parse(**kwargs):
        captured.update(kwargs)
        return FakeParsed()
    monkeypatch.setattr(tracker._client.responses, "parse", fake_parse)

    cid, parsed, raw = asyncio.run(tracker.track_response_grounded(
        "comment text", "We agree with commenters.", {"comment_id": "C1"}))
    assert cid == "C1"
    assert parsed["agency_decision"] == "accept"
    assert captured["tools"] == []          # web search disabled on grounded path
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_xai_grounded.py -v`
Expected: FAIL — `ImportError: cannot import name 'GROUNDED_RESPONSE_PROMPT'`.

- [ ] **Step 3a: Add `GROUNDED_RESPONSE_PROMPT`**

In `stratification_scripts/gemini_client.py`, after the `RESPONSE_TRACKING_PROMPT` triple-quoted string (after line 175), add:

```python
GROUNDED_RESPONSE_PROMPT = """You are a researcher studying U.S. federal notice-and-comment rulemaking.

You are given the agency's OWN response-to-comments discussion, taken from the Final Rule
preamble (SUPPLEMENTARY INFORMATION) and/or a docket "Response to Comments" document. This is
the primary, authoritative source. Agencies respond to comment THEMES in aggregate, so you must
MATCH this specific comment to the relevant theme.

YOUR JOB (per-comment):
1. Find where, if anywhere, the provided text addresses the concern raised in THIS comment.
2. Classify the agency's disposition of THIS comment's request: accept / reject / partial.

COMMENT DETAILS:
- Comment ID: {comment_id}
- Federal Register Document: {document_number}
- Agency: {agency}
- Commenter Type: {commenter_type}
- Submission Date: {submission_date}

COMMENT TEXT:
{full_comment_text}

AGENCY RESPONSE-TO-COMMENTS TEXT (primary source — use ONLY this; do not use outside knowledge or web search):
{grounded_text}

CLASSIFICATION RULES:
- response_found: "yes" only if the provided text addresses THIS comment's concern (or its theme).
  "no" if the provided text does NOT address it — do NOT borrow the agency's disposition of a
  different topic and apply it here. "uncertain" only if genuinely ambiguous.
- agency_decision (only when response_found="yes"):
  - "accept" = agency adopted the request / changed the rule in the requested direction (minor caveats still "accept").
  - "reject" = agency kept the provision / declined (acknowledging a small point is still "reject").
  - "partial" = ONLY when the comment raised multiple separable issues and the agency accepted some, rejected others.
  - "uncertain" = the text addresses it but disposition is unclear.
- If response_found is "no" or "uncertain", set agency_decision to "uncertain".

OUTPUT: valid JSON matching the schema.
- response_text: the SPECIFIC passage from the provided text that responds to this comment (not a generic summary), or "N/A".
- response_location: the matched section heading, or "N/A".
- reasoning: 1-2 sentences.
"""
```

- [ ] **Step 3b: Add grounded methods to `XAIResponseTracker`**

In `stratification_scripts/xai_response_client.py`, add an import near the top (with the existing gemini_client import):

```python
from stratification_scripts.gemini_client import (
    AgencyResponse,
    RESPONSE_TRACKING_PROMPT,
    GROUNDED_RESPONSE_PROMPT,
)
```

Add these methods to `XAIResponseTracker` (after `_track_response_sync`, before `track_response`):

```python
    def _track_grounded_sync(self, comment_text, grounding_text, metadata):
        max_c = 20000
        if len(comment_text) > max_c:
            comment_text = comment_text[:max_c] + "\n\n[... truncated ...]"
        prompt = GROUNDED_RESPONSE_PROMPT.format(
            comment_id=metadata.get("comment_id", "N/A"),
            document_number=metadata.get("document_number", "N/A"),
            agency=metadata.get("agency", "N/A"),
            commenter_type=metadata.get("commenter_type", "N/A"),
            submission_date=metadata.get("submission_date", "N/A"),
            full_comment_text=comment_text,
            grounded_text=grounding_text,
        )
        comment_id = metadata.get("comment_id", "unknown")
        backoff = 2.0
        for attempt in range(self.max_retries):
            try:
                response = self._client.responses.parse(
                    model=self.model,
                    instructions=("You are studying U.S. rulemaking. Classify the agency's disposition of "
                                  "THIS public comment using ONLY the provided primary-source text. Do not "
                                  "search the web. Return structured JSON."),
                    input=prompt,
                    tools=[],  # NO web search on the grounded path
                    text_format=AgencyResponse,
                )
                parsed = response.output_parsed
                if parsed is not None:
                    return (comment_id, parsed.normalized(), (response.output_text or "")[:500] or "OK_JSON")
                if response.output_text:
                    try:
                        manual = AgencyResponse.model_validate_json(response.output_text)
                        return (comment_id, manual.normalized(), response.output_text[:500])
                    except Exception:
                        pass
                return (comment_id, AgencyResponse(response_found="uncertain", agency_decision="uncertain",
                        response_text="N/A", response_location="N/A",
                        reasoning="Empty model response").normalized(), "EMPTY_RESPONSE")
            except Exception as e:  # noqa: BLE001
                if attempt >= self.max_retries - 1 or not self._is_retryable_error(e):
                    return (comment_id, AgencyResponse(response_found="uncertain", agency_decision="uncertain",
                            response_text="N/A", response_location="N/A",
                            reasoning=f"API error: {type(e).__name__}").normalized(), f"ERROR: {e}")
                time.sleep(backoff); backoff = min(backoff * 2, 60.0)
        return (comment_id, AgencyResponse(response_found="uncertain", agency_decision="uncertain",
                response_text="N/A", response_location="N/A",
                reasoning="API retries exhausted").normalized(), "ERROR: retries_exhausted")

    async def track_response_grounded(self, comment_text, grounding_text, metadata, semaphore=None):
        async def do():
            return await asyncio.to_thread(self._track_grounded_sync, comment_text, grounding_text, metadata)
        if semaphore:
            async with semaphore:
                return await do()
        return await do()

    async def track_grounded_batch(self, items, max_concurrency=50):
        """items: list of (comment_text, grounding_text, metadata)."""
        if not items:
            return []
        sem = asyncio.Semaphore(max_concurrency)
        tasks = [asyncio.create_task(self.track_response_grounded(c, g, m, sem)) for c, g, m in items]
        out = []
        for t in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Tracking responses (xAI grounded)"):
            out.append(await t)
        return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_xai_grounded.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add stratification_scripts/gemini_client.py stratification_scripts/xai_response_client.py tests/test_xai_grounded.py
git commit -m "feat(responses): grounded (no-search) Grok classify + GROUNDED_RESPONSE_PROMPT"
```

---

## Task 5: Wire grounded routing into `track_responses`

**Files:**
- Modify: `stratification_scripts/config.py` (add two fields)
- Modify: `stratification_scripts/makeup/track_responses.py`

- [ ] **Step 1: Add config fields**

In `stratification_scripts/config.py`, inside `PipelineConfig` under "Response tracking settings" (after line 400 `response_provider`), add:

```python
    # Primary-source grounding (Final Rule preamble) for Tier 1
    enable_primary_source_grounding: bool = True
    grounded_max_chars: int = 100_000
```

- [ ] **Step 2: Add the cache-builder + routing helper (test-first)**

Create `tests/test_grounded_routing.py`:

```python
import polars as pl
from stratification_scripts.makeup.track_responses import build_grounded_cache

def test_build_grounded_cache_maps_doc_to_final_rule(monkeypatch):
    # comment doc 2024-NPRM links to final rule 2024-FINAL; extractor returns evidence
    fr = pl.DataFrame({"document_number": ["2024-NPRM"],
                       "final_rule_document_number": ["2024-FINAL"]})
    import stratification_scripts.makeup.track_responses as tr
    monkeypatch.setattr(tr, "_fetch_final_rule_xml", lambda dn: "<SUPLINF><HD SOURCE=\"HD1\">Response to Comments</HD><P>" + ("x " * 2000) + "we agree.</P></SUPLINF>")
    cache = build_grounded_cache(["2024-NPRM"], fr, grounded_max_chars=100000)
    assert "2024-NPRM" in cache
    assert "we agree" in cache["2024-NPRM"].grounded_text
```

Run: `python -m pytest tests/test_grounded_routing.py -v` → FAIL (`build_grounded_cache` undefined).

- [ ] **Step 3: Implement cache + routing in `track_responses.py`**

Add imports at top of `stratification_scripts/makeup/track_responses.py`:

```python
from stratification_scripts.makeup.fr_response_extractor import (
    extract_response_section, ResponseExtract,
)
from stratification_scripts.federal_register.client import FederalRegisterClient
```

Add these module-level functions (near `extract_full_comment_text`):

```python
def _fetch_final_rule_xml(final_doc_number: str) -> Optional[str]:
    """Fetch structured full_text_xml for a Final Rule (own client; short-lived)."""
    client = FederalRegisterClient(max_retries=4, sleep_between=0.3)
    try:
        return client.fetch_document_full_text_xml(final_doc_number)
    finally:
        client.close()


def build_grounded_cache(comment_doc_numbers, df_fr, grounded_max_chars=100_000):
    """Map each comment document_number -> ResponseExtract for its linked Final Rule.

    A comment's doc may itself be the final rule, or link to one via
    final_rule_document_number. Returns {doc_number: ResponseExtract} only for
    docs with a usable grounded_text. Best-effort: failures are skipped.
    """
    # doc -> final-rule doc number
    link = {}
    cols = df_fr.columns
    for row in df_fr.select([c for c in ["document_number", "final_rule_document_number", "doc_type"] if c in cols]).iter_rows(named=True):
        dn = str(row.get("document_number") or "")
        frn = str(row.get("final_rule_document_number") or "").strip()
        if frn and frn.lower() not in ("none", "null", ""):
            link[dn] = frn
        elif str(row.get("doc_type") or "") == "Rule":
            link[dn] = dn  # the doc is itself a final rule
    targets = {dn: link[dn] for dn in set(map(str, comment_doc_numbers)) if dn in link}
    # fetch+extract unique final rules once
    extract_by_final: dict = {}
    for frn in sorted(set(targets.values())):
        xml = _fetch_final_rule_xml(frn)
        if not xml:
            continue
        ext = extract_response_section(xml)
        if ext.grounded_text:
            extract_by_final[frn] = ext
    return {dn: extract_by_final[frn] for dn, frn in targets.items() if frn in extract_by_final}
```

Run: `python -m pytest tests/test_grounded_routing.py -v` → PASS.

- [ ] **Step 4: Route grounded-vs-fallback inside `track_responses_for_year` Tier 1**

In `track_responses_for_year`, after `df_to_process`/`comments_to_process` is built and before the `process_responses_async` call (around line 729-759), insert grounded routing. Replace the single `asyncio.run(process_responses_async(...))` block with:

```python
        grounded_cache = {}
        if getattr(config, "enable_primary_source_grounding", True) and provider == "xai" and fr_csv.exists():
            doc_nums = [str(c.get("document_number")) for c in comments_to_process]
            try:
                grounded_cache = build_grounded_cache(doc_nums, df_fr, config.grounded_max_chars)
                logger.info(f"Primary-source grounding: {len(grounded_cache)} comments have a Final-Rule response section")
            except Exception as e:
                logger.warning(f"Grounded cache build failed, falling back to web search: {e}")

        grounded_items, grounded_meta, fallback = [], [], []
        for c in comments_to_process:
            dn = str(c.get("document_number"))
            ext = grounded_cache.get(dn)
            if ext is not None:
                full_text = extract_full_comment_text(c, client=regs_client, max_pages=config.max_comment_pages)
                meta = {"comment_id": str(c.get("comment_id")),
                        "document_number": dn, "agency": str(c.get("agency", "N/A")),
                        "commenter_type": str(c.get("category", "N/A")),
                        "submission_date": str(c.get("posted_date", "N/A"))}
                grounded_items.append((full_text, ext.grounded_text, meta))
                grounded_meta.append((c, ext))
            else:
                fallback.append(c)

        try:
            if grounded_items:
                logger.info(f"Tier 1 grounded path: {len(grounded_items)} comments (no web search)")
                gres = asyncio.run(tracker.track_grounded_batch(grounded_items, max_concurrency=max_concurrency))
                _save_grounded_results(responses_csv, gres, grounded_meta, weight_map, df_raw if has_deduplication else None)
            if fallback:
                logger.info(f"Tier 1 web-search fallback: {len(fallback)} comments")
                asyncio.run(process_responses_async(
                    tracker, fallback, responses_csv, max_concurrency, regs_client,
                    config.max_comment_pages, df_raw if has_deduplication else None, batch_size, weight_map))
        finally:
            if regs_client:
                regs_client.close()
```

> Move the `regs_client` initialization (lines ~737-745) ABOVE this block so it is available for `extract_full_comment_text`.

Add the grounded-results saver near `save_responses_incremental`:

```python
def _save_grounded_results(responses_csv, results, grounded_meta, weight_map, df_comments):
    by_id = {str(c.get("comment_id")): (c, ext) for c, ext in grounded_meta}
    rows = []
    for comment_id, parsed, _raw in results:
        c, ext = by_id.get(comment_id, ({}, None))
        src = "rtc_doc" if (ext and ext.method == "rtc_doc") else "fr_preamble"
        rows.append({
            "comment_id": comment_id,
            "document_number": str(c.get("document_number") or "N/A"),
            "agency": str(c.get("agency") or "N/A"),
            "response_found": parsed.get("response_found", "uncertain"),
            "agency_decision": parsed.get("agency_decision", "uncertain"),
            "response_text": parsed.get("response_text", "N/A"),
            "response_location": parsed.get("response_location", "N/A"),
            "reasoning": parsed.get("reasoning", "N/A"),
            "processed_at": datetime.now().isoformat(),
            "model": tracker_model_name(),
            "comment_text_length": 0,
            "has_attachment": bool(c.get("attachment_text")),
            "lifecycle_stage": str(c.get("lifecycle_stage") or "UNKNOWN"),
            "rin": str(c.get("rin") or "N/A"),
            "response_sample_weight": weight_map.get(comment_id, 1.0) if weight_map else 1.0,
            "response_source": src,
            "response_citation": (ext.matched_header if ext else "") or "",
            "rtc_document_id": "",
        })
    save_responses_incremental(responses_csv, rows, df_comments)
```

> `tracker_model_name()` — replace with the tracker's model string in scope (the function has access to `tracker.model`; inline `tracker.model` if simpler). Also add `"response_source": "web_search"` (+ empty `response_citation`/`rtc_document_id`) to the row dict built in `process_responses_async` so both paths share a schema.

- [ ] **Step 5: Run the full unit suite**

Run: `python -m pytest tests/ -v`
Expected: all pass (no live API/network — routing logic covered by `test_grounded_routing.py`; the live path is exercised in Task 6).

- [ ] **Step 6: Commit**

```bash
git add stratification_scripts/config.py stratification_scripts/makeup/track_responses.py tests/test_grounded_routing.py
git commit -m "feat(responses): route Tier 1 to primary-source grounding with web-search fallback"
```

---

## Task 6: Mid-N end-to-end verification (not a unit test)

**Files:** none (verification only).

- [ ] **Step 1: Run Step 5 on a small real slice**

Pre-req: a year with existing upstream CSVs (2024 present in repo). Source keys: `source prd/keys.txt` (or export from it). Then:

Run: `python -m stratification_scripts.makeup.track_responses --year 2024 --provider xai --limit 40 --verbose`
Expected: log lines `Primary-source grounding: N comments have a Final-Rule response section`, `Tier 1 grounded path: …`, `Tier 1 web-search fallback: …`; no crash; `agency_responses_2024.csv` gains `response_source`/`response_citation`/`rtc_document_id`.

- [ ] **Step 2: Sanity-check the output distribution**

Run:
```bash
python -c "import polars as pl; d=pl.read_csv('stratification_scripts/makeup/data/agency_responses_2024.csv', infer_schema_length=None); print(d['response_source'].value_counts()); print(d.filter(pl.col('response_source')=='fr_preamble').select('response_found','agency_decision','response_citation').head(10))"
```
Expected: a mix of `fr_preamble` and `web_search`; grounded rows carry non-empty citations and per-comment decisions (NOT all identical within a docket — confirms comment-level granularity).

- [ ] **Step 3: Adversarial spot-check via workflow (optional, recommended under ultracode)**

Dispatch a small verification workflow: sample 10 grounded rows, have agents confirm the `agency_decision` is supported by the cited `response_text` for that specific comment. Confirm no aggregate-bleed (different comments on the same docket can have different decisions).

- [ ] **Step 4: Commit any fixes; mark P4 done.**

---

## Self-review

- **Spec coverage:** §2 approach → Tasks 4-5; §2.5 comment-level → grounded prompt (Task 4) + per-comment routing/save (Task 5) + Task 6 Step 2 check; §3.1 extractor → Task 1; §3.1 RTC → Task 2; §3.2 XML fetch → Task 3; §3.3 grounded tracker → Task 4; §3.4 prompt → Task 4; §3.5 routing → Task 5; §4 schema columns → Task 5 `_save_grounded_results` + web_search row; §5 algorithm → Task 1 (ported, validated); §7 fallback ladder → Task 5 routing (grounded miss → fallback) + extractor `no_preamble`; §8 testing → Tasks 1-5 + Task 6.
- **Placeholders:** `tracker_model_name()` is explicitly flagged to inline `tracker.model`. No other TBDs.
- **Type consistency:** `ResponseExtract` fields (`grounded_text`, `method`, `matched_header`, `found_response_hd`, `suplinf_len`, `grounded_len`) used consistently across Tasks 1/5; `track_grounded_batch` takes `(comment_text, grounding_text, metadata)` tuples consistent with `_save_grounded_results` pairing via `grounded_meta`.
- **RTC body download** is deferred (Task 2 finds titles; full body-text fetch + `rtc_doc` sourcing is a follow-up — `response_source='rtc_doc'` is wired but populated only once body download is added; sparse, non-blocking).
