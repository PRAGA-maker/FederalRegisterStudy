from stratification_scripts.resolution.models import (
    CandidateDocument, Channel, CommentRef, Relevance, ResponseEvidence,
    RuleClass, Status,
)
from stratification_scripts.resolution.resolver import (
    DocumentResolver, merge_candidates, qualifying_candidates,
)

REF = CommentRef(
    comment_id="DOT-OST-2024-0090-0049", comment_date="2024-09-23",
    source_document="2024-18496", agency="Transportation Department",
    rins=("2105-AF05",), docket_id="Docket No. DOT-OST-2024-0090",
    packet_final_document="2025-02747",
)

FCC_DOC = {"document_number": "2025-02747", "title": "Radio Broadcasting Services",
           "type": "Rule", "action": "Final rule.", "publication_date": "2025-02-19",
           "agencies": [{"name": "Federal Communications Commission"}],
           "regulation_id_numbers": [], "docket_ids": ["DA 25-120"]}

DOT_FINAL = {"document_number": "2024-29990", "title": "T", "type": "Rule",
             "action": "Final rule.", "publication_date": "2024-12-18",
             "agencies": [{"name": "Transportation Department"}, {"name": None}],
             "regulation_id_numbers": ["2105-AF05"],
             "docket_ids": ["Docket No. DOT-OST-2024-0090"]}

DOT_NPRM = {"document_number": "2024-18496", "title": "N", "type": "Proposed Rule",
            "action": "Notice of proposed rulemaking (NPRM).",
            "publication_date": "2024-08-22",
            "agencies": [{"name": "Transportation Department"}],
            "regulation_id_numbers": ["2105-AF05"],
            "docket_ids": ["Docket No. DOT-OST-2024-0090"]}

DENSE_XML = (
    "<RULE><SUPLINF>"
    + ("Comment: Several commenters argued this. Response: We disagree with the "
       "commenters and are adopting the provision. In response to comment, the "
       "agency considered it. ") * 20
    + "</SUPLINF></RULE>"
)


class FakeFR:
    def __init__(self):
        self.xml_calls = []
        self.docs = {d["document_number"]: d for d in (FCC_DOC, DOT_FINAL, DOT_NPRM)}

    def fetch_document_details(self, document_number, enrich_identifiers=True):
        return self.docs.get(document_number)

    def fetch_document_full_text_xml(self, document_number):
        self.xml_calls.append(document_number)
        return DENSE_XML

    def search_by_rin(self, rin):
        return [DOT_FINAL, DOT_NPRM]

    def search_by_docket(self, docket_id):
        return []

    def search_full_text(self, identifier):
        return []


class FakeRegInfo:
    def fetch_unified_agenda(self, rin):
        return {"rin": rin, "stage": "FINAL", "timetable": [
            {"action": "NPRM", "date": "2024-08-22", "date_raw": "", "citation": ""},
            {"action": "FINAL RULE", "date": "2024-12-18", "date_raw": "", "citation": ""},
        ], "withdrawn": False}


def test_dot_row_resolves_found_via_rin_search_despite_bad_packet_link():
    fr = FakeFR()
    result = DocumentResolver(fr_client=fr, reginfo_client=FakeRegInfo()).resolve(REF)
    assert result.status is Status.FOUND
    assert result.absence_reason is None
    by_num = {c.document_number: c for c in result.candidates}
    assert by_num["2025-02747"].relevance is Relevance.AGENCY_MISMATCH
    assert by_num["2024-29990"].discovered_by is Channel.RIN_SEARCH
    assert by_num["2024-29990"].response_evidence is ResponseEvidence.STRONG


def test_fetch_policy_skips_non_qualifying_candidates():
    fr = FakeFR()
    DocumentResolver(fr_client=fr, reginfo_client=FakeRegInfo()).resolve(REF)
    assert fr.xml_calls == ["2024-29990"]


def test_channels_run_records_the_envelope_per_row():
    result = DocumentResolver(fr_client=FakeFR(), reginfo_client=FakeRegInfo()).resolve(REF)
    assert set(result.channels_run) == set(Channel)
    assert result.channels_run[Channel.DOCKET_SEARCH] == "ok"


def test_merge_keeps_the_most_precise_channel():
    def make(channel):
        return CandidateDocument(
            document_number="2024-29990", publication_date="2024-12-18", type="Rule",
            action="Final rule.", title="t", agency_names=("Transportation Department",),
            rule_class=RuleClass.FINAL, rins=("2105-AF05",), docket_id=None,
            discovered_by=channel, postdates_comment=True, relevance=Relevance.MATCH,
        )

    merged = merge_candidates([[make(Channel.FULLTEXT_SEARCH)], [make(Channel.RIN_SEARCH)]])
    assert len(merged) == 1
    assert merged[0].discovered_by is Channel.RIN_SEARCH


def test_qualifying_candidates_are_all_returned_earliest_first():
    result = DocumentResolver(fr_client=FakeFR(), reginfo_client=FakeRegInfo()).resolve(REF)
    quals = qualifying_candidates(result)
    assert [c.document_number for c in quals] == ["2024-29990"]


def test_agenda_failure_is_recorded_and_forces_unknown_on_absence():
    class NoAgenda(FakeRegInfo):
        def fetch_unified_agenda(self, rin):
            return None

    class EmptyFR(FakeFR):
        def fetch_document_details(self, document_number, enrich_identifiers=True):
            return None

        def search_by_rin(self, rin):
            return []

    result = DocumentResolver(fr_client=EmptyFR(), reginfo_client=NoAgenda()).resolve(REF)
    assert result.status is Status.UNKNOWN
    assert result.agenda is not None and result.agenda.ok is False
    assert result.channels_run[Channel.AGENDA].startswith("failed:")


def test_empty_comment_date_short_circuits_to_unknown():
    undated = CommentRef(comment_id="X-1", comment_date="", source_document="d",
                         agency="Transportation Department", rins=("2105-AF05",),
                         docket_id=None, packet_final_document=None)
    result = DocumentResolver(fr_client=FakeFR(), reginfo_client=FakeRegInfo()).resolve(undated)
    assert result.status is Status.UNKNOWN
    assert result.absence_reason is None
    assert result.candidates == []
    assert all(s == "skipped:no comment date" for s in result.channels_run.values())


def test_resolver_runs_fulltext_and_marks_it_ok():
    result = DocumentResolver(fr_client=FakeFR(), reginfo_client=FakeRegInfo()).resolve(REF)
    assert result.channels_run[Channel.FULLTEXT_SEARCH] == "ok"
