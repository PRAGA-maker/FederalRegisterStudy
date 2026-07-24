# tests/test_resolution_routing.py
from stratification_scripts.makeup.resolution_routing import ENVELOPE_VERSION, ref_from_row


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
