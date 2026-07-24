"""Routing between the pipeline's comment rows and the resolution layer.

Pure functions only — no I/O, no clients. track_responses injects the resolver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from stratification_scripts.resolution import CommentRef, ResolutionResult, Status
from stratification_scripts.resolution.resolver import qualifying_candidates

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


@dataclass
class RoutedOutcome:
    """Where one comment goes after resolution: grounded LLM read, typed absence, or unknown."""
    kind: str                      # "grounded" | "absent" | "unknown"
    result: ResolutionResult
    candidate: object = None       # CandidateDocument for grounded, else None
    extract: object = None         # ResponseExtract for grounded, else None


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


def partition_by_resolution(comments, resolver):
    """Resolve every comment once and split by route.

    Returns (grounded, absent, unknown); each item is (comment_row, RoutedOutcome).
    The resolver's cross-row cache makes repeated documents cheap; repeated
    comment_ids (shouldn't happen, but joins have surprised us before — F22)
    are resolved once and reuse the outcome. Rows with an empty/missing
    comment_id are never memoized together — they'd otherwise collapse onto a
    shared "" key and each subsequent one would silently inherit the first
    row's outcome instead of being resolved on its own merits.
    """
    grounded, absent, unknown = [], [], []
    outcomes = {}
    for row in comments:
        cid = str(row.get("comment_id") or "")
        if cid:
            if cid not in outcomes:
                outcomes[cid] = route_resolution(resolver.resolve(ref_from_row(row)), resolver.cache)
            outcome = outcomes[cid]
        else:
            outcome = route_resolution(resolver.resolve(ref_from_row(row)), resolver.cache)
        {"grounded": grounded, "absent": absent, "unknown": unknown}[outcome.kind].append((row, outcome))
    return grounded, absent, unknown


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
