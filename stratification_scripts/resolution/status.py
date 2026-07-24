"""Three-valued status derivation."""

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
    """Whether a candidate could carry the response to this comment."""
    return (
        candidate.rule_class is RuleClass.FINAL
        and candidate.postdates_comment
        and candidate.relevance is Relevance.MATCH
    )


def all_channels_clean(channels_run: Mapping[Channel, str]) -> bool:
    """True only when every declared channel ran without failure or skip."""
    return all(channels_run.get(channel) == CHANNEL_OK for channel in Channel)


def _blocks_absence(candidates: Iterable[CandidateDocument]) -> bool:
    return any(
        candidate.rule_class is not RuleClass.FINAL
        and candidate.postdates_comment
        and candidate.relevance is Relevance.MATCH
        and candidate.response_evidence is ResponseEvidence.STRONG
        for candidate in candidates
    )


def _absence_reason(
    candidates: List[CandidateDocument], agenda: AgendaStatus
) -> Optional[AbsenceReason]:
    structural = [
        candidate for candidate in candidates
        if candidate.rule_class in _NO_VENUE_CLASSES
        and candidate.relevance is Relevance.MATCH
        and candidate.response_evidence is not ResponseEvidence.STRONG
    ]
    if structural and not agenda.withdrawn:
        return AbsenceReason.NO_VENUE_POSSIBLE
    if agenda.final_rule_undetermined or (agenda.stage or "").upper() == "LONG_TERM":
        return AbsenceReason.NO_FINAL_RULE_PLANNED
    pending = [
        candidate for candidate in candidates
        if candidate.rule_class in _NOT_YET_CLASSES
        and candidate.relevance is Relevance.MATCH
    ]
    if pending:
        return AbsenceReason.RESPONSE_NOT_YET_PUBLISHED
    return None


def derive_status(
    candidates: List[CandidateDocument],
    agenda: Optional[AgendaStatus],
    channels_run: Mapping[Channel, str],
) -> Tuple[Status, Optional[AbsenceReason]]:
    """Derive status and absence reason from candidates, agenda, and envelope."""
    qualifying = [candidate for candidate in candidates if qualifies(candidate)]
    if any(
        candidate.response_evidence is not ResponseEvidence.NONE
        for candidate in qualifying
    ):
        return Status.FOUND, None
    if qualifying:
        return Status.UNKNOWN, None
    if _blocks_absence(candidates):
        return Status.UNKNOWN, None
    if not all_channels_clean(channels_run):
        return Status.UNKNOWN, None
    if agenda is None or not agenda.ok:
        return Status.UNKNOWN, None
    reason = _absence_reason(candidates, agenda)
    if reason is None:
        return Status.UNKNOWN, None
    return Status.CONFIDENTLY_ABSENT, reason
