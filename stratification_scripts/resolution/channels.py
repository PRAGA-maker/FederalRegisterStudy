"""The five discovery channels, ordered precise to wide."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from ..federal_register.client import normalize_docket_id
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
    """Build one candidate from an FR API document record."""
    rins = tuple(str(rin) for rin in doc.get("regulation_id_numbers") or [] if rin)
    dockets = [str(docket) for docket in doc.get("docket_ids") or [] if docket]
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
            discovered_by=discovered_by,
            agency_names=agency_names,
            rins=rins,
            docket_id=docket_id,
            ref=ref,
        ),
    )


def run_packet_link(ref: CommentRef, cache) -> ChannelOutcome:
    """Channel 1: validate and return the packet's existing final-document link."""
    if not ref.packet_final_document:
        return ChannelOutcome([], "skipped:no packet link")
    details = cache.details(ref.packet_final_document)
    if not details:
        return ChannelOutcome(
            [], f"failed:details fetch {ref.packet_final_document}"
        )
    return ChannelOutcome(
        [candidate_from_fr_doc(
            details, ref=ref, discovered_by=Channel.PACKET_LINK
        )],
        STATE_OK,
    )


def _collect(
    docs: Optional[List[dict]], ref: CommentRef, channel: Channel
) -> List[CandidateDocument]:
    seen = set()
    candidates: List[CandidateDocument] = []
    for doc in docs or []:
        number = str(doc.get("document_number") or "")
        if not number or number in seen:
            continue
        seen.add(number)
        candidates.append(
            candidate_from_fr_doc(doc, ref=ref, discovered_by=channel)
        )
    return candidates


def run_rin_search(ref: CommentRef, fr_client) -> ChannelOutcome:
    """Channel 2: all documents filed under the comment's RINs."""
    rins = [rin for rin in dict.fromkeys(ref.rins) if rin]
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
    """Channel 3: all documents filed under the comment's docket id."""
    if not ref.docket_id:
        return ChannelOutcome([], "skipped:no docket")
    docs = fr_client.search_by_docket(ref.docket_id)
    if docs is None:
        return ChannelOutcome([], f"failed:docket search {ref.docket_id}")
    return ChannelOutcome(_collect(docs, ref, Channel.DOCKET_SEARCH), STATE_OK)


def fulltext_identifiers(ref: CommentRef) -> List[str]:
    """Identifier queries for channel 5, never subject terms."""
    identifiers: List[str] = []
    for value in [ref.docket_id, normalize_docket_id(ref.docket_id), *ref.rins]:
        token = (value or "").strip()
        if token and " " not in token and token not in identifiers:
            identifiers.append(token)
    return identifiers


def run_fulltext_search(ref: CommentRef, fr_client) -> ChannelOutcome:
    """Channel 5: full-text search keyed only on lineage identifiers."""
    identifiers = fulltext_identifiers(ref)
    if not identifiers:
        return ChannelOutcome([], "skipped:no identifiers")
    docs: List[dict] = []
    for identifier in identifiers:
        found = fr_client.search_full_text(identifier)
        if found is None:
            return ChannelOutcome(
                [], f"failed:fulltext search {identifier}"
            )
        docs.extend(found)
    return ChannelOutcome(
        _collect(docs, ref, Channel.FULLTEXT_SEARCH), STATE_OK
    )
