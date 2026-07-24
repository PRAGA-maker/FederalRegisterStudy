"""Build CommentRefs from a frozen snapshot or goldset seed packet."""

from __future__ import annotations

from typing import List, Optional, Sequence

import polars as pl

from .. import config
from .models import CommentRef


def _rins(row: dict) -> tuple:
    raw = str(row.get("rin_all") or row.get("rin") or "")
    parts = [part.strip() for part in raw.replace(";", ",").split(",")]
    return tuple(dict.fromkeys(
        part for part in parts
        if part and part.lower() not in ("none", "null")
    ))


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
    """Join snapshot comment and FR tables into CommentRefs."""
    base = config.get_frozen_snapshot_path(snapshot_id)
    comments = pl.read_csv(
        base / f"makeup/data/comments_raw_{year}.csv", infer_schema_length=0
    ).select(["comment_id", "document_number", "posted_date", "receive_date"])
    fr = pl.read_csv(
        base / f"output/federal_register_{year}_comments.csv",
        infer_schema_length=0,
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
    """Build CommentRefs for exactly the rows in a goldset labeling packet."""
    packet_path = config.get_goldset_seed_path(seed_id) / "labeling_packet.csv"
    packet = pl.read_csv(packet_path, infer_schema_length=0)
    refs: List[CommentRef] = []
    for row in packet.iter_rows(named=True):
        refs.append(CommentRef(
            comment_id=str(row["comment_id"]),
            comment_date="",
            source_document=str(row.get("document_number") or ""),
            agency=str(row.get("agency") or ""),
            rins=_rins(row),
            docket_id=_clean(row.get("docket_id")),
            packet_final_document=_clean(row.get("final_rule_document_number")),
        ))
    return refs
