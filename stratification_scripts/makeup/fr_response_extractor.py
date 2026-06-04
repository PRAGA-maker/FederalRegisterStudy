"""Extract the primary-source agency 'Response to Comments' discussion from a
Federal Register Final Rule's structured full_text_xml, to feed an LLM that
classifies PER-COMMENT agency dispositions.

Algorithm validated on a 37-rule, ~20-agency, 2014+2024 heterogeneity sample
(prd/govinfo_probe/extract_proto.py + adversarial verification workflow):
100% primary-source preamble coverage; 19/19 present-response rules captured.

The output `grounded_text` is *evidence* for the classifier, NOT an answer:
the classifier must match each individual comment to the relevant theme-response
within it. See docs/superpowers/specs/2026-06-04-primary-source-response-extraction-design.md
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from stratification_scripts.logging_utils import get_logger

logger = get_logger(__name__)

GROUND_CAP = 100_000   # max chars of grounded evidence fed to the LLM
POINTER_MIN = 2_500    # a matched response head shorter than this may be just a pointer

# Header phrases that introduce an agency's response-to-comments discussion.
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
# Loose "this head is about comments" test — keeps contiguous comment-by-topic blocks.
COMMENTISH = re.compile(r"\bcomment", re.I)
# Comment-discussion density signal: mentions + agency-disposition verbs.
DENSITY_KW = re.compile(
    r"\b(comment(s|er|ers)?|in\s+response|we\s+(agree|disagree|adopt(ed)?|decline[d]?|considered"
    r"|are\s+(not\s+)?(adopting|persuaded))|the\s+(agency|department|commission|commenter))\b", re.I)
BOILER = re.compile(
    r"^\s*(AGENCY|ACTION|SUMMARY|DATES|ADDRESSES|FOR FURTHER|SUPPLEMENTARY|EFFECTIVE|TABLE OF CONTENTS)\b", re.I)

# Standalone Response-to-Comments document titles in regulations.gov dockets.
RTC_TITLE = re.compile(
    r"(response\s+to\s+comment|comment.{0,12}response|\bRTC\b|summary\s+of\s+(public\s+)?comment)", re.I)
RTC_MAX_CHARS = 60_000


@dataclass
class ResponseExtract:
    """Primary-source evidence for per-comment response classification."""
    grounded_text: str                # what to feed the LLM ("" if no preamble)
    method: str                       # response_hd | response_hd_to_full | suplinf_full | comment_density | no_preamble
    matched_header: Optional[str]     # the response-section head, if one matched
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
    """For preambles larger than `cap`, keep the HD-delimited chunks with the
    highest comment/response keyword density (concentrates the budget on the
    actual adjudication text wherever it lives), restored to reading order."""
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
        picked.append((start, txt))
        total += len(txt)
        if total >= cap:
            break
    picked.sort(key=lambda c: c[0])
    return ("\n\n[...]\n\n".join(t for _, t in picked))[:cap]


def extract_response_section(xml: str) -> ResponseExtract:
    """Isolate the agency response-to-comments evidence from FR full_text_xml.

    Container = <SUPLINF> (or <PREAMB> fallback). Within it, locate the response
    head and slice to the next same-or-higher-level non-comment head (keeps
    comment-by-topic blocks). Substantial section -> use it; else whole preamble
    if it fits GROUND_CAP; else comment-density selection for giant rules.
    """
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
        lvl = _hd_level(m.group(1))
        txt = _clean(m.group(2))
        if not txt or BOILER.match(txt):
            continue
        if RESP_HD.search(txt):
            found = True
            matched_header = txt
            start = m.start()
            end = len(body)
            for j in range(i + 1, len(hds)):
                if _hd_level(hds[j].group(1)) <= lvl and not COMMENTISH.search(_clean(hds[j].group(2))):
                    end = hds[j].start()
                    break
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


def is_rtc_title(title: str) -> bool:
    """True if a docket document title looks like a standalone Response-to-Comments doc."""
    return bool(title and RTC_TITLE.search(title))


def find_docket_rtc_documents(docket_id: str, regs_client, max_chars: int = RTC_MAX_CHARS) -> List[dict]:
    """Return [{document_id, title, text}] for docket 'Supporting & Related Material'
    documents whose titles look like standalone Response-to-Comments docs. Sparse
    (~1/8 EPA dockets); best-effort, never raises. Body text is left empty here
    (the caller downloads it on demand)."""
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
        if is_rtc_title(title):
            out.append({"document_id": d.get("id"), "title": title, "text": ""})
    return out
