from stratification_scripts.resolution.models import (
    AbsenceReason, AgendaStatus, CandidateDocument, Channel, Relevance,
    ResponseEvidence, RuleClass, Status,
)
from stratification_scripts.resolution.status import (
    CHANNEL_OK, all_channels_clean, derive_status, qualifies,
)

ALL_CLEAN = {c: CHANNEL_OK for c in Channel}


def cand(**kw) -> CandidateDocument:
    base = dict(
        document_number="2024-00001", publication_date="2024-12-01", type="Rule",
        action="Final rule.", title="t", agency_names=("Agency",),
        rule_class=RuleClass.FINAL, rins=(), docket_id=None,
        discovered_by=Channel.RIN_SEARCH, postdates_comment=True,
        relevance=Relevance.MATCH, response_evidence=ResponseEvidence.STRONG,
    )
    base.update(kw)
    return CandidateDocument(**base)


def agenda(**kw) -> AgendaStatus:
    base = dict(rin="1234-AB56", stage="FINAL", timetable=[],
                final_rule_undetermined=False, withdrawn=False,
                fetched_at="2026-07-23T00:00:00", ok=True)
    base.update(kw)
    return AgendaStatus(**base)


def test_qualification_ignores_response_evidence():
    assert qualifies(cand(response_evidence=ResponseEvidence.NONE)) is True
    assert qualifies(cand(rule_class=RuleClass.DIRECT_FINAL)) is False
    assert qualifies(cand(postdates_comment=False)) is False
    assert qualifies(cand(relevance=Relevance.AGENCY_MISMATCH)) is False


def test_found_needs_only_its_own_evidence():
    status, reason = derive_status([cand()], agenda(), {Channel.PACKET_LINK: CHANNEL_OK})
    assert status is Status.FOUND and reason is None


def test_dot_regression_weak_header_still_found():
    c = cand(response_evidence=ResponseEvidence.STRONG, response_header_matched=False)
    status, _ = derive_status([c], agenda(), ALL_CLEAN)
    assert status is Status.FOUND


def test_weak_evidence_on_a_qualifying_candidate_is_found_not_absent():
    status, _ = derive_status([cand(response_evidence=ResponseEvidence.WEAK)], agenda(), ALL_CLEAN)
    assert status is Status.FOUND


def test_qualifying_but_unreadable_is_unknown_never_absent():
    status, reason = derive_status(
        [cand(response_evidence=ResponseEvidence.NONE)], agenda(), ALL_CLEAN)
    assert status is Status.UNKNOWN and reason is None


def test_deferred_variant_with_real_text_blocks_absence():
    ifc = cand(rule_class=RuleClass.INTERIM_FINAL,
               action="Interim final action with comment period.",
               response_evidence=ResponseEvidence.STRONG,
               response_header_matched=False)
    status, reason = derive_status([ifc], agenda(final_rule_undetermined=True), ALL_CLEAN)
    assert status is Status.UNKNOWN and reason is None


def test_failed_channel_forces_unknown():
    channels = dict(ALL_CLEAN)
    channels[Channel.FULLTEXT_SEARCH] = "failed:HTTP 503"
    status, reason = derive_status([], agenda(final_rule_undetermined=True), channels)
    assert status is Status.UNKNOWN and reason is None


def test_zero_result_channel_is_clean_not_a_failure():
    assert all_channels_clean(ALL_CLEAN) is True
    assert all_channels_clean({**ALL_CLEAN, Channel.DOCKET_SEARCH: "skipped:no docket"}) is False


def test_missing_agenda_forces_unknown():
    status, reason = derive_status([], None, ALL_CLEAN)
    assert status is Status.UNKNOWN and reason is None
    status, reason = derive_status([], agenda(ok=False), ALL_CLEAN)
    assert status is Status.UNKNOWN and reason is None


def test_blm_direct_final_is_no_venue_possible():
    dfr = cand(rule_class=RuleClass.DIRECT_FINAL, action="Direct final rule.",
               postdates_comment=False, response_evidence=ResponseEvidence.NONE)
    status, reason = derive_status([dfr], agenda(stage="COMPLETED"), ALL_CLEAN)
    assert status is Status.CONFIDENTLY_ABSENT
    assert reason is AbsenceReason.NO_VENUE_POSSIBLE


def test_direct_final_carrying_real_responses_is_not_no_venue_possible():
    dfr = cand(rule_class=RuleClass.DIRECT_FINAL, action="Direct final rule.",
               postdates_comment=True, response_evidence=ResponseEvidence.STRONG)
    status, reason = derive_status([dfr], agenda(stage="COMPLETED"), ALL_CLEAN)
    assert status is Status.UNKNOWN and reason is None


def test_withdrawn_direct_final_is_not_no_venue_possible():
    dfr = cand(rule_class=RuleClass.DIRECT_FINAL, postdates_comment=False,
               response_evidence=ResponseEvidence.NONE)
    status, reason = derive_status([dfr], agenda(withdrawn=True, stage="COMPLETED"), ALL_CLEAN)
    assert status is not Status.CONFIDENTLY_ABSENT or reason is not AbsenceReason.NO_VENUE_POSSIBLE


def test_fbi_nprm_only_is_response_not_yet_published():
    nprm = cand(rule_class=RuleClass.PROPOSED,
                action="Notice of proposed rulemaking (NPRM).",
                postdates_comment=False, response_evidence=ResponseEvidence.NONE)
    status, reason = derive_status([nprm], agenda(stage="FINAL"), ALL_CLEAN)
    assert status is Status.CONFIDENTLY_ABSENT
    assert reason is AbsenceReason.RESPONSE_NOT_YET_PUBLISHED


def test_epa_tbd_agenda_outranks_not_yet_published():
    nprm = cand(rule_class=RuleClass.PROPOSED, action="Proposed rule.",
                postdates_comment=False, response_evidence=ResponseEvidence.NONE)
    status, reason = derive_status(
        [nprm], agenda(stage="LONG_TERM", final_rule_undetermined=True), ALL_CLEAN)
    assert status is Status.CONFIDENTLY_ABSENT
    assert reason is AbsenceReason.NO_FINAL_RULE_PLANNED


def test_no_candidates_and_no_corroboration_is_unknown():
    status, reason = derive_status([], agenda(stage="FINAL"), ALL_CLEAN)
    assert status is Status.UNKNOWN and reason is None
