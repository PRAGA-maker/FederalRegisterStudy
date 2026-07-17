from __future__ import annotations

import polars as pl

from stratification_scripts.goldset import packet


def _sampled_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "label_row_id": ["a1", "b2", "c3"],
            "comment_id": ["EPA-2024-0001-0001", "FAA-2024-0002-0005", "SBA-2024-0007-0003"],
            "document_number": ["2024-00001", "2024-00002", "2024-00003"],
            "agency": ["EPA", "FAA", "SBA"],
            "rin": ["2060-AV12", "2120-AL55", "3245-AH99"],
            "response_source": ["web_search", "fr_preamble", "web_search"],
            "response_sample_weight": [17.4, 7.5, 12.0],
            "overlap_candidate": [True, False, True],
            # hidden verdicts that must NOT reach the packet:
            "response_found": ["no", "no", "no"],
            "agency_decision": ["", "", ""],
            "response_text": ["secret", "secret", "secret"],
        }
    )


def _context_fixture():
    comments_raw = pl.DataFrame(
        {
            "comment_id": ["EPA-2024-0001-0001", "FAA-2024-0002-0005", "SBA-2024-0007-0003"],
            "comment_text": ["please regulate", "safety concern", "small biz impact"],
            "organization": ["NRDC", "", "Main St LLC"],
            "submitter_type": ["Organization", "Individual", "Organization"],
        }
    )
    fr = pl.DataFrame(
        {
            "document_number": ["2024-00001", "2024-00002", "2024-00003"],
            "title": ["Rule A", "Rule B", "Rule C"],
            "docket_id": ["Docket ID EPA-2024-0001", "FAR Case 2019-015", "Docket ID SBA-2024-0007"],
            "final_action_citation": ["89 FR 102448", "", "89 FR 55000"],
            "final_rule_document_number": ["2024-90001", "", "2024-90003"],
        }
    )
    return comments_raw, fr


def test_links_layer_correctly():
    linked = packet.build_links(_sampled_fixture())
    assert linked["rin_url"].to_list() == [
        "https://www.federalregister.gov/r/2060-AV12",
        "https://www.federalregister.gov/r/2120-AL55",
        "https://www.federalregister.gov/r/3245-AH99",
    ]
    assert linked["nprm_url"][0] == "https://www.federalregister.gov/d/2024-00001"
    assert linked["comment_url"][0] == "https://www.regulations.gov/comment/EPA-2024-0001-0001"


def test_packet_excludes_every_forbidden_column():
    p, _ = packet.build_packet_and_key(
        _sampled_fixture(), snapshot_id="S", context=_context_fixture()
    )
    for col in packet.FORBIDDEN_IN_PACKET:
        assert col not in p.columns, f"forbidden column leaked into packet: {col}"


def test_packet_has_inputs_and_empty_label_columns():
    p, _ = packet.build_packet_and_key(
        _sampled_fixture(), snapshot_id="S", context=_context_fixture()
    )
    assert "comment_text" in p.columns and "rin_url" in p.columns
    for col in packet.LABEL_COLUMNS:
        assert col in p.columns
        assert p[col].fill_null("").to_list() == ["", "", ""]  # empty for labeler to fill


def test_final_rule_url_blank_exactly_when_missing():
    p, _ = packet.build_packet_and_key(
        _sampled_fixture(), snapshot_id="S", context=_context_fixture()
    )
    urls = dict(zip(p["comment_id"].to_list(), p["final_rule_url"].to_list()))
    assert urls["EPA-2024-0001-0001"] == "https://www.federalregister.gov/d/2024-90001"
    assert urls["FAA-2024-0002-0005"] == ""  # final_rule_document_number was blank


def test_no_url_constructed_from_docket_id():
    p, _ = packet.build_packet_and_key(
        _sampled_fixture(), snapshot_id="S", context=_context_fixture()
    )
    # docket_id ships as prose; no column should turn it into a URL
    assert "docket_url" not in p.columns
    joined_urls = " ".join(
        v for c in p.columns if c.endswith("_url") for v in p[c].fill_null("").to_list()
    )
    assert "FAR Case" not in joined_urls and "Docket ID" not in joined_urls


def test_key_and_packet_join_is_total():
    p, key = packet.build_packet_and_key(
        _sampled_fixture(), snapshot_id="S", context=_context_fixture()
    )
    assert sorted(p["label_row_id"].to_list()) == sorted(key["label_row_id"].to_list())
    assert key["label_row_id"].n_unique() == p.height
    # key carries what grading needs, packet does not
    assert "response_source" in key.columns and "response_source" not in p.columns
    assert "response_sample_weight" in key.columns
