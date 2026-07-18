from __future__ import annotations

from stratification_scripts.rtc_parser import responses

TXT = """Comment Excerpt from Commenter 52
The public relies on EPA.
Individual Response: Please see Discussion on General Comments and Contaminant Groups. EPA agrees here.
Comment Excerpt from Commenter 99
An orphan excerpt with no response.
Comment Excerpt from Commenter 61
Lowercase discussion word here.
Individual Response: Please see discussion on Chemical Technical Support Documents.
Comment Excerpt from Commenter 71
A comment with a response but no cross-reference.
Individual Response: EPA understands the concerns raised and continues its work.
"""


def test_blocks_capture_excerpt_clause_supplemental():
    blocks = {b.commenter_number: b for b in responses.split_comment_blocks(TXT)}
    assert "public relies on EPA" in blocks[52].comment_excerpt
    assert blocks[52].raw_topic_clause.strip() == "General Comments and Contaminant Groups"
    assert "EPA agrees here" in blocks[52].individual_response_supplemental
    assert blocks[52].has_individual_response is True


def test_orphan_excerpt_has_no_individual_response():
    blocks = {b.commenter_number: b for b in responses.split_comment_blocks(TXT)}
    assert blocks[99].has_individual_response is False
    assert blocks[99].raw_topic_clause == ""
    assert len(blocks) == 4  # off-by-one does not swallow the orphan


def test_lowercase_discussion_word_still_resolves_clause():
    blocks = {b.commenter_number: b for b in responses.split_comment_blocks(TXT)}
    assert blocks[61].raw_topic_clause.strip() == "Chemical Technical Support Documents"


def test_individual_response_without_cross_reference():
    blocks = {b.commenter_number: b for b in responses.split_comment_blocks(TXT)}
    assert blocks[71].has_individual_response is True
    assert blocks[71].raw_topic_clause == ""  # no "Please see Discussion" clause
    assert "EPA understands the concerns" in blocks[71].individual_response_supplemental
