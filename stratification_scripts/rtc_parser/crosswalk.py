"""Assemble the final per-comment crosswalk records.

Joins comment blocks (excerpt + cross-ref clause) to Exhibit 2 commenters (for
Document ID + submitter identity) and to the Agency Topic Discussions (for
disposition text), resolving each cross-reference to a canonical topic.
"""

from __future__ import annotations

from stratification_scripts.rtc_parser import topics
from stratification_scripts.rtc_parser.models import Commenter, CommentRecord
from stratification_scripts.rtc_parser.responses import CommentBlock


def assemble(
    commenters: list[Commenter],
    blocks: list[CommentBlock],
    discussions: dict[str, str],
) -> list[CommentRecord]:
    """Build one CommentRecord per comment block, in block order."""
    by_number = {c.number: c for c in commenters}
    records: list[CommentRecord] = []
    for b in blocks:
        c = by_number.get(b.commenter_number)
        refs = topics.resolve_refs(b.raw_topic_clause) if b.has_individual_response else []
        topic_discussions = {
            r.canonical: discussions[r.canonical]
            for r in refs
            if r.resolved and r.canonical in discussions
        }
        records.append(
            CommentRecord(
                commenter_number=b.commenter_number,
                document_id=c.document_id if c else None,
                first_name=c.first_name if c else "",
                last_name=c.last_name if c else "",
                organization=c.organization if c else "",
                comment_excerpt=b.comment_excerpt,
                has_individual_response=b.has_individual_response,
                topic_refs=refs,
                individual_response_supplemental=b.individual_response_supplemental,
                topic_discussions=topic_discussions,
            )
        )
    return records
