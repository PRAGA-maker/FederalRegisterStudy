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
    rin_url, nprm_url, final_url, comment_url = [], [], [], []
    for r in sampled.iter_rows(named=True):
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
    """Left-join after deduping `right` on the key; assert no row fan-out.

    An unguarded many-to-many join on a non-unique key silently fabricates rows.
    Both keys are unique in the 2024 snapshot, so this guards against a future
    dirty input, not a present bug.
    """
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
    the pinned snapshot. Joins are one-to-one guarded (see _guarded_join).
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
