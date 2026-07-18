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


def test_resolve_refs_abbrev_multi_and_unresolved():
    refs = topics.resolve_refs("General Comments, PFAS, and 1,4-Dioxane")
    by_raw = {r.raw: r for r in refs}
    assert by_raw["General Comments"].canonical == "General Comments"
    assert by_raw["PFAS"].canonical == "Per- and Polyfluoroalkyl substances (PFAS)"
    assert by_raw["PFAS"].resolved is True
    # out-of-list mention is kept, flagged, never force-matched
    assert by_raw["1,4-Dioxane"].resolved is False
    assert by_raw["1,4-Dioxane"].canonical is None
    assert len(refs) == 3  # nothing dropped


def test_resolve_refs_joins_linebreak_split_topic():
    refs = topics.resolve_refs("Draft CCL\n5-Microbes")
    assert len(refs) == 1
    assert refs[0].canonical == "Draft CCL 5-Microbes"
    assert refs[0].resolved is True


def test_resolve_refs_dbps_abbrev_and_empty_clause():
    assert topics.resolve_refs("DBPs")[0].canonical == "Disinfection Byproducts (DBPs)"
    assert topics.resolve_refs("") == []
