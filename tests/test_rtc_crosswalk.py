from __future__ import annotations

from stratification_scripts.rtc_parser.models import Commenter, CommentRecord, TopicRef


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
