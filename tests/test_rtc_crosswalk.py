from __future__ import annotations

from stratification_scripts.rtc_parser import crosswalk
from stratification_scripts.rtc_parser.models import Commenter, CommentRecord, TopicRef
from stratification_scripts.rtc_parser.responses import CommentBlock


def test_models_construct():
    c = Commenter(54, "EPA-HQ-OW-2018-0594-0054", "", "", "Anonymous")
    t = TopicRef(raw="PFAS", canonical="Per- and Polyfluoroalkyl substances (PFAS)", resolved=True)
    r = CommentRecord(
        commenter_number=54,
        document_id=c.document_id,
        first_name="",
        last_name="",
        organization="Anonymous",
        comment_excerpt="excerpt",
        has_individual_response=True,
        topic_refs=[t],
        individual_response_supplemental="",
        topic_discussions={},
    )
    assert c.number == 54
    assert t.resolved and t.canonical.endswith("(PFAS)")
    assert r.topic_refs[0].raw == "PFAS"


def test_assemble_joins_docid_topics_and_discussion():
    commenters = [Commenter(52, "EPA-HQ-OW-2018-0594-0052", "A", "B", "Org")]
    blocks = [
        CommentBlock(52, "excerpt", True, "General Comments and PFAS", "supp"),
        CommentBlock(99, "orphan", False, "", ""),
    ]
    disc = {
        "General Comments": "GC text",
        "Per- and Polyfluoroalkyl substances (PFAS)": "PFAS text",
    }
    recs = {r.commenter_number: r for r in crosswalk.assemble(commenters, blocks, disc)}
    assert recs[52].document_id == "EPA-HQ-OW-2018-0594-0052"
    assert recs[52].organization == "Org"
    assert {t.raw for t in recs[52].topic_refs} == {"General Comments", "PFAS"}
    assert recs[52].topic_discussions["General Comments"] == "GC text"
    assert recs[52].topic_discussions["Per- and Polyfluoroalkyl substances (PFAS)"] == "PFAS text"
    # orphan: no commenter match -> document_id None, no topics
    assert recs[99].document_id is None
    assert recs[99].topic_refs == []
    assert recs[99].has_individual_response is False


def test_assemble_keeps_unresolved_ref_without_discussion():
    blocks = [CommentBlock(60, "x", True, "1,4-Dioxane", "")]
    recs = crosswalk.assemble([], blocks, {})
    assert recs[0].topic_refs[0].resolved is False
    assert recs[0].topic_discussions == {}  # unresolved -> no discussion attached
