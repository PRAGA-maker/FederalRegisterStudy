# tests/test_resolution_routing.py
from stratification_scripts.makeup.resolution_routing import (
    ENVELOPE_VERSION,
    RoutedOutcome,  # noqa: F401 -- imported to assert the type is exported
    partition_by_resolution,
    ref_from_row,
    route_resolution,
    typed_fields,
)
from stratification_scripts.resolution import (
    AbsenceReason,
    CandidateDocument,
    Channel,
    Relevance,
    ResolutionResult,
    ResponseEvidence,
    RuleClass,
    Status,
)


def _row(**kw):
    base = dict(
        comment_id="DOT-OST-2024-0090-0049", document_number="2024-18496",
        agency="Transportation Department", posted_date="2024-09-23T14:00:00Z",
        receive_date="2024-09-22", rin="2105-AF05", rin_all="2105-AF05;2105-ZZ99",
        docket_id="Docket No. DOT-OST-2024-0090",
        final_rule_document_number="2025-02747",
    )
    base.update(kw)
    return base


def test_envelope_version_is_v1():
    assert ENVELOPE_VERSION == "v1"


def test_ref_from_full_row():
    ref = ref_from_row(_row())
    assert ref.comment_id == "DOT-OST-2024-0090-0049"
    assert ref.comment_date == "2024-09-23"          # posted_date wins, date part only
    assert ref.source_document == "2024-18496"
    assert ref.rins == ("2105-AF05", "2105-ZZ99")    # rin_all split, deduped, order kept
    assert ref.docket_id == "Docket No. DOT-OST-2024-0090"
    assert ref.packet_final_document == "2025-02747"


def test_receive_date_backfills_missing_posted():
    ref = ref_from_row(_row(posted_date=None))
    assert ref.comment_date == "2024-09-22"


def test_undated_row_yields_empty_comment_date():
    ref = ref_from_row(_row(posted_date=None, receive_date=""))
    assert ref.comment_date == ""                     # resolver guard handles it


def test_rin_fallback_when_no_rin_all():
    ref = ref_from_row(_row(rin_all=None))
    assert ref.rins == ("2105-AF05",)


def test_none_and_null_strings_are_cleaned():
    ref = ref_from_row(_row(final_rule_document_number="None", docket_id="null", rin="none", rin_all=None))
    assert ref.packet_final_document is None
    assert ref.docket_id is None
    assert ref.rins == ()


def _cand(**kw):
    base = dict(
        document_number="2024-29990", publication_date="2024-12-18", type="Rule",
        action="Final rule.", title="t", agency_names=("Transportation Department",),
        rule_class=RuleClass.FINAL, rins=("2105-AF05",), docket_id=None,
        discovered_by=Channel.RIN_SEARCH, postdates_comment=True,
        relevance=Relevance.MATCH, response_evidence=ResponseEvidence.STRONG,
    )
    base.update(kw)
    return CandidateDocument(**base)


def _result(status, reason=None, candidates=(), channels=None):
    return ResolutionResult(
        comment_id="X-1", comment_date="2024-09-23", source_document="d",
        status=status, absence_reason=reason, candidates=list(candidates),
        agenda=None, channels_run=channels or {c: "ok" for c in Channel},
        resolved_at="2026-07-24T00:00:00",
    )


class FakeCache:
    def __init__(self, extracts):
        self._e = extracts

    def extract(self, document_number):
        return self._e.get(document_number)


class FakeExtract:
    grounded_text = "Comment: x Response: we agree"
    matched_header = "Comments and Responses"
    found_response_hd = True


def test_found_routes_to_grounded_with_first_qualifying_candidate():
    res = _result(Status.FOUND, candidates=[_cand()])
    out = route_resolution(res, FakeCache({"2024-29990": FakeExtract()}))
    assert out.kind == "grounded"
    assert out.candidate.document_number == "2024-29990"
    assert out.extract.grounded_text


def test_found_with_unreadable_extract_degrades_to_unknown_not_absent():
    res = _result(Status.FOUND, candidates=[_cand()])
    out = route_resolution(res, FakeCache({}))          # cache miss
    assert out.kind == "unknown"


def test_confidently_absent_routes_to_absent():
    res = _result(Status.CONFIDENTLY_ABSENT, reason=AbsenceReason.NO_VENUE_POSSIBLE,
                  candidates=[_cand(rule_class=RuleClass.DIRECT_FINAL, postdates_comment=False,
                                    response_evidence=ResponseEvidence.NONE)])
    out = route_resolution(res, FakeCache({}))
    assert out.kind == "absent"


def test_unknown_routes_to_unknown():
    res = _result(Status.UNKNOWN)
    assert route_resolution(res, FakeCache({})).kind == "unknown"


def test_typed_fields_grounded():
    res = _result(Status.FOUND, candidates=[_cand()])
    fields = typed_fields(route_resolution(res, FakeCache({"2024-29990": FakeExtract()})))
    assert fields["resolution_status"] == "FOUND"
    assert fields["absence_reason"] == ""
    assert fields["envelope_version"] == "v1"
    assert fields["resolved_document_number"] == "2024-29990"
    assert fields["discovered_by"] == "RIN_SEARCH"
    assert "PACKET_LINK:ok" in fields["resolution_channels"]


def test_typed_fields_absent_carries_reason():
    res = _result(Status.CONFIDENTLY_ABSENT, reason=AbsenceReason.RESPONSE_NOT_YET_PUBLISHED)
    fields = typed_fields(route_resolution(res, FakeCache({})))
    assert fields["resolution_status"] == "CONFIDENTLY_ABSENT"
    assert fields["absence_reason"] == "RESPONSE_NOT_YET_PUBLISHED"
    assert fields["resolved_document_number"] == ""


def test_unknown_never_renders_as_no():
    # The invariant, tested at the field level: unknown kind carries UNKNOWN status,
    # and (Task 4) only CONFIDENTLY_ABSENT rows may write response_found="no".
    res = _result(Status.UNKNOWN)
    fields = typed_fields(route_resolution(res, FakeCache({})))
    assert fields["resolution_status"] == "UNKNOWN"


class FakeResolver:
    """Resolves by comment_id via a fixed table; counts calls for the dedup test."""
    def __init__(self, table, cache):
        self.table, self.cache, self.calls = table, cache, []

    def resolve(self, ref):
        self.calls.append(ref.comment_id)
        return self.table[ref.comment_id]


def test_partition_routes_each_comment_and_shares_the_cache():
    found = _result(Status.FOUND, candidates=[_cand()])
    absent = _result(Status.CONFIDENTLY_ABSENT, reason=AbsenceReason.NO_FINAL_RULE_PLANNED)
    unknown = _result(Status.UNKNOWN)
    resolver = FakeResolver(
        {"A": found, "B": absent, "C": unknown},
        FakeCache({"2024-29990": FakeExtract()}),
    )
    comments = [
        _row(comment_id="A"), _row(comment_id="B"), _row(comment_id="C"),
    ]
    grounded, absent_rows, unknown_rows = partition_by_resolution(comments, resolver)
    assert [c["comment_id"] for c, _ in grounded] == ["A"]
    assert [c["comment_id"] for c, _ in absent_rows] == ["B"]
    assert [c["comment_id"] for c, _ in unknown_rows] == ["C"]
    assert grounded[0][1].extract.grounded_text


def test_partition_resolves_each_comment_id_once():
    res = _result(Status.UNKNOWN)
    resolver = FakeResolver({"A": res}, FakeCache({}))
    partition_by_resolution([_row(comment_id="A"), _row(comment_id="A")], resolver)
    assert resolver.calls == ["A"]        # second row reuses the first resolution
