from stratification_scripts import config
from stratification_scripts.resolution.models import (
    AbsenceReason, AgendaStatus, CandidateDocument, Channel, CommentRef,
    Relevance, ResolutionResult, ResponseEvidence, RuleClass, Status,
)


def test_resolution_paths_are_project_siblings():
    assert config.get_resolution_dir() == config.get_project_root() / "resolution"
    assert config.get_resolution_run_path("r1") == config.get_project_root() / "resolution" / "r1"


def test_enum_values_are_the_spec_strings():
    assert Channel.PACKET_LINK.value == "PACKET_LINK"
    assert Channel.FULLTEXT_SEARCH.value == "FULLTEXT_SEARCH"
    assert RuleClass.CONFIRMATION_OF_EFFECTIVE_DATE.value == "CONFIRMATION_OF_EFFECTIVE_DATE"
    assert Status.CONFIDENTLY_ABSENT.value == "CONFIDENTLY_ABSENT"
    assert AbsenceReason.NO_VENUE_POSSIBLE.value == "NO_VENUE_POSSIBLE"
    assert Relevance.AGENCY_MISMATCH.value == "AGENCY_MISMATCH"
    assert ResponseEvidence.WEAK.value == "WEAK"


def test_result_round_trips_to_json_dict():
    ref = CommentRef(
        comment_id="NOAA-NMFS-2023-0125-0016", comment_date="2024-03-22",
        source_document="2024-01120", agency="Commerce Department, National Oceanic",
        rins=("0648-BM40",), docket_id="NOAA-NMFS-2023-0125",
        packet_final_document="2024-15931",
    )
    cand = CandidateDocument(
        document_number="2024-15931", publication_date="2024-07-19", type="Rule",
        action="Final rule.", title="t", agency_names=("Commerce Department",),
        rule_class=RuleClass.FINAL, rins=("0648-BM40",), docket_id="NOAA-NMFS-2023-0125",
        discovered_by=Channel.PACKET_LINK, postdates_comment=True,
        relevance=Relevance.MATCH, response_evidence=ResponseEvidence.STRONG,
        response_header_matched=True, response_section_ref="Comments and Responses",
    )
    result = ResolutionResult(
        comment_id=ref.comment_id, comment_date=ref.comment_date,
        source_document=ref.source_document, status=Status.FOUND, absence_reason=None,
        candidates=[cand],
        agenda=AgendaStatus(rin="0648-BM40", stage="COMPLETED", timetable=[],
                            final_rule_undetermined=False, withdrawn=False,
                            fetched_at="2026-07-23T00:00:00", ok=True),
        channels_run={Channel.PACKET_LINK: "ok"},
        resolved_at="2026-07-23T00:00:00",
    )
    d = result.to_dict()
    assert d["status"] == "FOUND"
    assert d["absence_reason"] is None
    assert d["candidates"][0]["rule_class"] == "FINAL"
    assert d["channels_run"]["PACKET_LINK"] == "ok"
