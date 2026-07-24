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
    comment_date: str
    source_document: str
    agency: str
    rins: Tuple[str, ...]
    docket_id: Optional[str]
    packet_final_document: Optional[str]


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
    ok: bool

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
