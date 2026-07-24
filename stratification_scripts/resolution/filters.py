"""Chronology and relevance filters."""

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
    """True when the candidate was published on or after the comment date."""
    pub, com = _iso(publication_date), _iso(comment_date)
    if pub is None or com is None:
        return False
    return pub >= com


def normalize_agency(name: Optional[str]) -> str:
    if not name:
        return ""
    return _SPACE.sub(" ", _PUNCT.sub(" ", str(name).lower())).strip()


def agency_matches(
    candidate_agency_names: Sequence[Optional[str]], ref_agency: str
) -> bool:
    """True when any candidate agency overlaps the comment's agency string."""
    ref_parts = [normalize_agency(p) for p in str(ref_agency or "").split(",")]
    ref_parts = [p for p in ref_parts if p]
    cand_parts = [normalize_agency(n) for n in candidate_agency_names]
    cand_parts = [p for p in cand_parts if p]
    if not ref_parts or not cand_parts:
        return True
    for candidate in cand_parts:
        for reference in ref_parts:
            if candidate == reference or candidate in reference or reference in candidate:
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
    """Classify a candidate's relevance to the comment's rulemaking."""
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
        return Relevance.MATCH
    return Relevance.LINEAGE_MISMATCH
