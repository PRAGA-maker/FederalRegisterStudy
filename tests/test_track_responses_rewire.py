import polars as pl

from stratification_scripts.makeup.resolution_routing import RoutedOutcome, typed_fields
from stratification_scripts.makeup.track_responses import _save_resolved_norun_rows
from stratification_scripts.resolution import AbsenceReason, Channel, ResolutionResult, Status


def _res(status, reason=None):
    return ResolutionResult(
        comment_id="B-1", comment_date="2024-09-23", source_document="d",
        status=status, absence_reason=reason, candidates=[], agenda=None,
        channels_run={c: "ok" for c in Channel}, resolved_at="t",
    )


def _comment(cid="B-1"):
    return {"comment_id": cid, "document_number": "2024-11111", "agency": "EPA",
            "lifecycle_stage": "NPRM_CLOSED", "rin": "2060-AV81", "attachment_text": ""}


def test_absent_rows_write_typed_no_without_llm(tmp_path):
    csv = tmp_path / "responses.csv"
    outcome = RoutedOutcome("absent", _res(Status.CONFIDENTLY_ABSENT, AbsenceReason.NO_FINAL_RULE_PLANNED))
    _save_resolved_norun_rows(csv, [(_comment(), outcome)], weight_map=None, df_comments=None, kind="absent")
    df = pl.read_csv(str(csv), infer_schema_length=None)
    row = df.to_dicts()[0]
    assert row["response_found"] == "no"
    assert row["response_source"] == "resolver_envelope"
    assert row["resolution_status"] == "CONFIDENTLY_ABSENT"
    assert row["absence_reason"] == "NO_FINAL_RULE_PLANNED"
    assert row["envelope_version"] == "v1"
    assert row["model"] == "none:resolver"


def test_unknown_rows_write_uncertain_never_no(tmp_path):
    csv = tmp_path / "responses.csv"
    outcome = RoutedOutcome("unknown", _res(Status.UNKNOWN))
    _save_resolved_norun_rows(csv, [(_comment("C-1"), outcome)], weight_map=None, df_comments=None, kind="unknown")
    row = pl.read_csv(str(csv), infer_schema_length=None).to_dicts()[0]
    assert row["response_found"] == "uncertain"
    assert row["response_source"] == "resolver_unknown"
    assert row["resolution_status"] == "UNKNOWN"


def test_appends_compatibly_onto_legacy_schema(tmp_path):
    # A pre-rewire CSV lacks the typed columns; save must merge, not crash.
    csv = tmp_path / "responses.csv"
    pl.DataFrame([{
        "comment_id": "OLD-1", "document_number": "d", "agency": "a",
        "response_found": "yes", "agency_decision": "accept", "response_text": "t",
        "response_location": "l", "reasoning": "r", "processed_at": "p", "model": "m",
        "comment_text_length": 0, "has_attachment": False, "lifecycle_stage": "s",
        "rin": "r", "response_sample_weight": 1.0, "response_source": "fr_preamble",
        "response_citation": "", "rtc_document_id": "",
    }]).write_csv(str(csv))
    outcome = RoutedOutcome("absent", _res(Status.CONFIDENTLY_ABSENT, AbsenceReason.NO_VENUE_POSSIBLE))
    _save_resolved_norun_rows(csv, [(_comment(), outcome)], weight_map=None, df_comments=None, kind="absent")
    df = pl.read_csv(str(csv), infer_schema_length=None)
    assert len(df) == 2
    assert set(["resolution_status", "absence_reason", "envelope_version"]).issubset(df.columns)
