from __future__ import annotations

from stratification_scripts.rtc_parser import topics

TWO_TOPICS = """Agency Discussion on General Comments
Agency Topic Discussion:
EPA received many general comments. The Agency agrees.
Comments Received on General Comments
Comment Excerpt from Commenter 52
body
Agency Discussion on Length of CCL 5
Agency Topic Discussion:
EPA disagrees the list is too long.
Comment Excerpt from Commenter 71
body
"""


def test_split_topic_discussions_keys_and_text():
    d = topics.split_topic_discussions(TWO_TOPICS)
    assert set(d) == {"General Comments", "Length of CCL 5"}
    assert "The Agency agrees." in d["General Comments"]
    assert "too long" in d["Length of CCL 5"]
    # comment excerpts must not bleed into the discussion body
    assert "Comment Excerpt" not in d["General Comments"]
    assert "Comments Received on" not in d["General Comments"]


def test_canonical_topics_has_all_22():
    assert len(topics.CANONICAL_TOPICS) == 22
    assert "General Comments" in topics.CANONICAL_TOPICS
    assert "Per- and Polyfluoroalkyl substances (PFAS)" in topics.CANONICAL_TOPICS
