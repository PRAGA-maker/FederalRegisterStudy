from stratification_scripts.resolution.filters import (
    agency_matches, normalize_agency, postdates_comment, relevance_of,
)
from stratification_scripts.resolution.models import Channel, CommentRef, Relevance

DOT_REF = CommentRef(
    comment_id="DOT-OST-2024-0090-0049", comment_date="2024-09-23",
    source_document="2024-18496", agency="Transportation Department",
    rins=("2105-AF05",), docket_id="Docket No. DOT-OST-2024-0090",
    packet_final_document="2025-02747",
)

CMS_REF = CommentRef(
    comment_id="CMS-2024-0131-6043", comment_date="2024-12-03",
    source_document="2024-22765", agency="Health and Human Services Department",
    rins=("0938-AV34",), docket_id="CMS-1808-IFC", packet_final_document=None,
)


def test_postdates_comment():
    assert postdates_comment("2024-12-18", "2024-09-23") is True
    assert postdates_comment("2024-09-23", "2024-09-23") is True
    assert postdates_comment("2024-08-22", "2024-09-23") is False
    assert postdates_comment(None, "2024-09-23") is False
    assert postdates_comment("garbage", "2024-09-23") is False


def test_normalize_agency_strips_case_and_punctuation():
    assert normalize_agency("Health and Human Services Department") == "health and human services department"
    assert normalize_agency("  Transportation Department ") == "transportation department"
    assert normalize_agency(None) == ""


def test_agency_matches_tolerates_none_entries():
    assert agency_matches(["Transportation Department", None], "Transportation Department") is True


def test_agency_matches_on_multi_agency_ref_string():
    assert agency_matches(
        ["Commerce Department", "National Oceanic and Atmospheric Administration"],
        "Commerce Department, National Oceanic and Atmospheric Administration",
    ) is True


def test_agency_matches_parent_child():
    assert agency_matches(
        ["Health and Human Services Department", "Centers for Medicare & Medicaid Services"],
        "Health and Human Services Department",
    ) is True


def test_packet_link_to_wrong_agency_is_rejected():
    assert relevance_of(
        discovered_by=Channel.PACKET_LINK,
        agency_names=["Federal Communications Commission"],
        rins=[], docket_id="DA 25-120", ref=DOT_REF,
    ) is Relevance.AGENCY_MISMATCH


def test_packet_link_same_agency_wrong_lineage_is_rejected():
    assert relevance_of(
        discovered_by=Channel.PACKET_LINK,
        agency_names=["Transportation Department"],
        rins=["2105-ZZ99"], docket_id="Docket No. DOT-OST-2099-1111", ref=DOT_REF,
    ) is Relevance.LINEAGE_MISMATCH


def test_packet_link_matching_rin_passes():
    assert relevance_of(
        discovered_by=Channel.PACKET_LINK,
        agency_names=["Transportation Department"],
        rins=["2105-AF05"], docket_id=None, ref=DOT_REF,
    ) is Relevance.MATCH


def test_fulltext_hit_under_a_different_rin_and_docket_still_matches():
    assert relevance_of(
        discovered_by=Channel.FULLTEXT_SEARCH,
        agency_names=["Health and Human Services Department",
                      "Centers for Medicare & Medicaid Services"],
        rins=["0938-AV53"], docket_id="CMS-1809-F", ref=CMS_REF,
    ) is Relevance.MATCH


def test_fulltext_hit_from_another_agency_is_still_rejected():
    assert relevance_of(
        discovered_by=Channel.FULLTEXT_SEARCH,
        agency_names=["Federal Communications Commission"],
        rins=[], docket_id=None, ref=CMS_REF,
    ) is Relevance.AGENCY_MISMATCH


def test_unknown_candidate_agency_is_not_treated_as_a_mismatch():
    assert relevance_of(
        discovered_by=Channel.RIN_SEARCH,
        agency_names=[], rins=["2105-AF05"], docket_id=None, ref=DOT_REF,
    ) is Relevance.MATCH
