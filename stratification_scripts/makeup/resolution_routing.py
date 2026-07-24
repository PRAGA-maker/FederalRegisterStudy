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
