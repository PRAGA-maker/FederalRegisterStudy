"""Compose discovery channels, fetch evidence, and derive resolution status."""

from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List

from ..reginfo.client import has_undetermined_final_rule
from .cache import DocumentCache
from .channels import (
    ChannelOutcome, run_docket_search, run_fulltext_search, run_packet_link,
    run_rin_search,
)
from .evidence import response_evidence_from_extract
from .filters import relevance_of
from .models import (
    AgendaStatus, CandidateDocument, Channel, CommentRef, Relevance,
    ResolutionResult, RuleClass, Status,
)
from .status import derive_status, qualifies

FETCHABLE_CLASSES = {
    RuleClass.FINAL,
    RuleClass.INTERIM_FINAL,
    RuleClass.DIRECT_FINAL,
    RuleClass.CONFIRMATION_OF_EFFECTIVE_DATE,
}

_CHANNEL_PRECEDENCE = {
    Channel.PACKET_LINK: 0,
    Channel.RIN_SEARCH: 1,
    Channel.DOCKET_SEARCH: 2,
    Channel.AGENDA: 3,
    Channel.FULLTEXT_SEARCH: 4,
}


def _valid_comment_date(value: str) -> bool:
    try:
        date.fromisoformat((value or "")[:10])
        return True
    except ValueError:
        return False


def merge_candidates(
    groups: List[List[CandidateDocument]],
) -> List[CandidateDocument]:
    """Keep one candidate per document number from the most precise channel."""
    best: Dict[str, CandidateDocument] = {}
    for group in groups:
        for candidate in group:
            existing = best.get(candidate.document_number)
            if existing is None or (
                _CHANNEL_PRECEDENCE[candidate.discovered_by]
                < _CHANNEL_PRECEDENCE[existing.discovered_by]
            ):
                best[candidate.document_number] = candidate
    return sorted(
        best.values(),
        key=lambda candidate: (
            candidate.publication_date or "", candidate.document_number
        ),
    )


def qualifying_candidates(result: ResolutionResult) -> List[CandidateDocument]:
    """Return every qualifying candidate, earliest publication first."""
    return sorted(
        [candidate for candidate in result.candidates if qualifies(candidate)],
        key=lambda candidate: (
            candidate.publication_date or "", candidate.document_number
        ),
    )


class DocumentResolver:
    """Given a comment, return every document where a response could live."""

    def __init__(self, *, fr_client, reginfo_client=None, cache=None) -> None:
        self._fr = fr_client
        self._reginfo = reginfo_client
        self._cache = cache or DocumentCache(fr_client)

    @property
    def cache(self) -> DocumentCache:
        return self._cache

    def _fetch_agenda(self, ref: CommentRef) -> tuple[AgendaStatus, str]:
        fetched_at = datetime.now().isoformat()
        rin = ref.rins[0] if ref.rins else None
        if not rin:
            return (
                AgendaStatus(None, None, [], False, False, fetched_at, False),
                "skipped:no rin",
            )
        if self._reginfo is None:
            return (
                AgendaStatus(rin, None, [], False, False, fetched_at, False),
                "skipped:no reginfo client",
            )
        try:
            agenda = self._reginfo.fetch_unified_agenda(rin)
        except Exception as exc:  # noqa: BLE001
            return (
                AgendaStatus(rin, None, [], False, False, fetched_at, False),
                f"failed:agenda {type(exc).__name__}",
            )
        if not agenda:
            return (
                AgendaStatus(rin, None, [], False, False, fetched_at, False),
                "failed:agenda not found",
            )
        return (
            AgendaStatus(
                rin=rin,
                stage=agenda.get("stage"),
                timetable=list(agenda.get("timetable") or []),
                final_rule_undetermined=has_undetermined_final_rule(agenda),
                withdrawn=bool(agenda.get("withdrawn")),
                fetched_at=fetched_at,
                ok=True,
            ),
            "ok",
        )

    def _run_fulltext(self, ref: CommentRef) -> ChannelOutcome:
        return run_fulltext_search(ref, self._fr)

    def _apply_fetch_policy(
        self, candidates: List[CandidateDocument]
    ) -> None:
        for candidate in candidates:
            if candidate.relevance is not Relevance.MATCH:
                continue
            if not candidate.postdates_comment:
                continue
            if candidate.rule_class not in FETCHABLE_CLASSES:
                continue
            extract = self._cache.extract(candidate.document_number)
            candidate.response_evidence = response_evidence_from_extract(extract)
            if extract is not None:
                candidate.response_header_matched = extract.found_response_hd
                candidate.response_section_ref = extract.matched_header

    def resolve(self, ref: CommentRef) -> ResolutionResult:
        if not _valid_comment_date(ref.comment_date):
            return ResolutionResult(
                comment_id=ref.comment_id,
                comment_date=ref.comment_date,
                source_document=ref.source_document,
                status=Status.UNKNOWN,
                absence_reason=None,
                candidates=[],
                agenda=None,
                channels_run={
                    channel: "skipped:no comment date" for channel in Channel
                },
                resolved_at=datetime.now().isoformat(),
            )

        channels_run: Dict[Channel, str] = {}
        packet = run_packet_link(ref, self._cache)
        channels_run[Channel.PACKET_LINK] = packet.state
        rin = run_rin_search(ref, self._fr)
        channels_run[Channel.RIN_SEARCH] = rin.state
        docket = run_docket_search(ref, self._fr)
        channels_run[Channel.DOCKET_SEARCH] = docket.state
        agenda, agenda_state = self._fetch_agenda(ref)
        channels_run[Channel.AGENDA] = agenda_state
        fulltext = self._run_fulltext(ref)
        channels_run[Channel.FULLTEXT_SEARCH] = fulltext.state

        candidates = merge_candidates([
            packet.candidates, rin.candidates, docket.candidates,
            fulltext.candidates,
        ])
        for candidate in candidates:
            candidate.relevance = relevance_of(
                discovered_by=candidate.discovered_by,
                agency_names=candidate.agency_names,
                rins=candidate.rins,
                docket_id=candidate.docket_id,
                ref=ref,
            )
        self._apply_fetch_policy(candidates)
        status, reason = derive_status(candidates, agenda, channels_run)
        return ResolutionResult(
            comment_id=ref.comment_id,
            comment_date=ref.comment_date,
            source_document=ref.source_document,
            status=status,
            absence_reason=reason,
            candidates=candidates,
            agenda=agenda,
            channels_run=channels_run,
            resolved_at=datetime.now().isoformat(),
        )
