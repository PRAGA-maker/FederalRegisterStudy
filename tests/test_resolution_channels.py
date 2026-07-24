from stratification_scripts.resolution.cache import DocumentCache
from stratification_scripts.resolution.channels import (
    candidate_from_fr_doc, fulltext_identifiers, run_docket_search,
    run_fulltext_search, run_packet_link, run_rin_search,
)
from stratification_scripts.resolution.models import (
    Channel, CommentRef, Relevance, RuleClass,
)

DOT_REF = CommentRef(
    comment_id="DOT-OST-2024-0090-0049", comment_date="2024-09-23",
    source_document="2024-18496", agency="Transportation Department",
    rins=("2105-AF05",), docket_id="Docket No. DOT-OST-2024-0090",
    packet_final_document="2025-02747",
)

FCC_DOC = {
    "document_number": "2025-02747", "title": "Radio Broadcasting Services",
    "type": "Rule", "action": "Final rule.", "publication_date": "2025-02-19",
    "agencies": [{"name": "Federal Communications Commission"}],
    "regulation_id_numbers": [], "docket_ids": ["DA 25-120"],
}

DOT_FINAL = {
    "document_number": "2024-29990", "title": "Transportation for Individuals",
    "type": "Rule", "action": "Final rule.", "publication_date": "2024-12-18",
    "agencies": [{"name": "Transportation Department"}, {"name": None}],
    "regulation_id_numbers": ["2105-AF05"],
    "docket_ids": ["Docket No. DOT-OST-2024-0090"],
}

DOT_NPRM = {
    "document_number": "2024-18496", "title": "NPRM", "type": "Proposed Rule",
    "action": "Notice of proposed rulemaking (NPRM).", "publication_date": "2024-08-22",
    "agencies": [{"name": "Transportation Department"}],
    "regulation_id_numbers": ["2105-AF05"],
    "docket_ids": ["Docket No. DOT-OST-2024-0090"],
}


class FakeFR:
    def __init__(self, rin_docs=None, docket_docs=None, details=None):
        self._rin_docs, self._docket_docs = rin_docs, docket_docs
        self._details = details or {}

    def search_by_rin(self, rin):
        return self._rin_docs

    def search_by_docket(self, docket_id):
        return self._docket_docs

    def fetch_document_details(self, document_number, enrich_identifiers=True):
        return self._details.get(document_number)

    def fetch_document_full_text_xml(self, document_number):
        return None


def test_candidate_carries_provenance_and_classification():
    candidate = candidate_from_fr_doc(
        DOT_FINAL, ref=DOT_REF, discovered_by=Channel.RIN_SEARCH
    )
    assert candidate.document_number == "2024-29990"
    assert candidate.rule_class is RuleClass.FINAL
    assert candidate.discovered_by is Channel.RIN_SEARCH
    assert candidate.postdates_comment is True
    assert candidate.relevance is Relevance.MATCH
    assert candidate.agency_names == ("Transportation Department",)


def test_packet_link_to_another_agency_is_returned_but_marked_mismatch():
    cache = DocumentCache(FakeFR(details={"2025-02747": FCC_DOC}))
    outcome = run_packet_link(DOT_REF, cache)
    assert outcome.state == "ok"
    assert [c.document_number for c in outcome.candidates] == ["2025-02747"]
    assert outcome.candidates[0].relevance is Relevance.AGENCY_MISMATCH


def test_packet_link_absent_is_skipped_not_failed():
    ref = CommentRef(comment_id="x", comment_date="2024-01-01", source_document="d",
                     agency="A", rins=(), docket_id=None, packet_final_document=None)
    outcome = run_packet_link(ref, DocumentCache(FakeFR()))
    assert outcome.candidates == []
    assert outcome.state.startswith("skipped:")


def test_packet_link_fetch_failure_is_failed():
    outcome = run_packet_link(DOT_REF, DocumentCache(FakeFR(details={})))
    assert outcome.state.startswith("failed:")


def test_rin_search_returns_all_docs_under_the_rin():
    outcome = run_rin_search(DOT_REF, FakeFR(rin_docs=[DOT_FINAL, DOT_NPRM]))
    assert outcome.state == "ok"
    assert {c.document_number for c in outcome.candidates} == {"2024-29990", "2024-18496"}
    by_num = {c.document_number: c for c in outcome.candidates}
    assert by_num["2024-18496"].rule_class is RuleClass.PROPOSED
    assert by_num["2024-18496"].postdates_comment is False


def test_rin_search_failure_is_failed_not_empty():
    outcome = run_rin_search(DOT_REF, FakeFR(rin_docs=None))
    assert outcome.candidates == [] and outcome.state.startswith("failed:")


def test_docket_search_zero_results_is_ok():
    outcome = run_docket_search(DOT_REF, FakeFR(docket_docs=[]))
    assert outcome.candidates == [] and outcome.state == "ok"


def test_docket_search_without_a_docket_is_skipped():
    ref = CommentRef(comment_id="x", comment_date="2024-01-01", source_document="d",
                     agency="A", rins=("1234-AB56",), docket_id=None,
                     packet_final_document=None)
    outcome = run_docket_search(ref, FakeFR(docket_docs=[]))
    assert outcome.state.startswith("skipped:")


def test_multi_rin_search_is_deduplicated():
    ref = CommentRef(comment_id="x", comment_date="2024-01-01", source_document="d",
                     agency="Transportation Department",
                     rins=("2105-AF05", "2105-AF05"), docket_id=None,
                     packet_final_document=None)
    outcome = run_rin_search(ref, FakeFR(rin_docs=[DOT_FINAL]))
    assert len(outcome.candidates) == 1


CMS_REF = CommentRef(
    comment_id="CMS-2024-0131-6043", comment_date="2024-12-03",
    source_document="2024-22765", agency="Health and Human Services Department",
    rins=("0938-AV34",), docket_id="CMS-1808-IFC", packet_final_document=None,
)

CMS_LATER = {
    "document_number": "2025-14681",
    "title": "Medicare Program; Hospital Inpatient Prospective Payment Systems",
    "type": "Rule", "action": "Final rule.", "publication_date": "2025-08-04",
    "agencies": [{"name": "Health and Human Services Department"},
                 {"name": "Centers for Medicare & Medicaid Services"}],
    "regulation_id_numbers": ["0938-AV53"], "docket_ids": ["CMS-1809-F"],
}


def test_fulltext_identifiers_are_identifiers_only():
    identifiers = fulltext_identifiers(CMS_REF)
    assert "CMS-1808-IFC" in identifiers
    assert "0938-AV34" in identifiers
    assert all(len(identifier.split()) == 1 for identifier in identifiers)


def test_fulltext_recovers_the_cross_rin_response():
    class FakeFullText:
        def __init__(self):
            self.terms = []

        def search_full_text(self, identifier):
            self.terms.append(identifier)
            return [CMS_LATER] if identifier == "CMS-1808-IFC" else []

    outcome = run_fulltext_search(CMS_REF, FakeFullText())
    assert outcome.state == "ok"
    assert [c.document_number for c in outcome.candidates] == ["2025-14681"]
    assert outcome.candidates[0].relevance is Relevance.MATCH
    assert outcome.candidates[0].discovered_by is Channel.FULLTEXT_SEARCH


def test_fulltext_failure_on_any_identifier_is_failed():
    class Failing:
        def search_full_text(self, identifier):
            return None

    assert run_fulltext_search(CMS_REF, Failing()).state.startswith("failed:")


def test_fulltext_without_identifiers_is_skipped():
    ref = CommentRef(comment_id="x", comment_date="2024-01-01", source_document="d",
                     agency="A", rins=(), docket_id=None, packet_final_document=None)

    class Unused:
        def search_full_text(self, identifier):
            raise AssertionError("should not be called")

    assert run_fulltext_search(ref, Unused()).state.startswith("skipped:")
