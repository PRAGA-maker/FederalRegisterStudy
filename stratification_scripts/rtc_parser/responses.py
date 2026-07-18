"""Split Section 2 into per-comment blocks: excerpt + Individual Response.

Blocks are anchored on `Comment Excerpt from Commenter <N>`; each spans to the
next such anchor or to the next topic's `Agency Discussion on ...` header
(whichever comes first, so a block never absorbs the next topic's discussion).

An Individual Response may (a) carry a `Please see Discussion[s] on <topics>.`
cross-reference, (b) have supplemental text only with no cross-reference, or
(c) be absent entirely (the 114-vs-113 orphan). All three are represented
faithfully rather than misaligned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from stratification_scripts.rtc_parser.clean import strip_running_headers

_EXCERPT = re.compile(r"Comment Excerpt from Commenter (\d+)")
_TOPIC_BOUNDARY = re.compile(r"Agency Discussion on ")
_IR_MARKER = "Individual Response:"
# case-insensitive "discussion"; the clause runs to the first period.
_CLAUSE = re.compile(r"please see discussions? on (.*?)\.", re.IGNORECASE | re.DOTALL)


@dataclass
class CommentBlock:
    commenter_number: int
    comment_excerpt: str
    has_individual_response: bool
    raw_topic_clause: str
    individual_response_supplemental: str


def split_comment_blocks(text: str) -> list[CommentBlock]:
    """Return one CommentBlock per `Comment Excerpt from Commenter` anchor."""
    anchors = list(_EXCERPT.finditer(text))
    blocks: list[CommentBlock] = []
    for i, a in enumerate(anchors):
        start = a.end()
        end = anchors[i + 1].start() if i + 1 < len(anchors) else len(text)
        # Cut at the next topic boundary so trailing next-topic discussion prose
        # never leaks into this block's supplemental text.
        boundary = _TOPIC_BOUNDARY.search(text, start, end)
        if boundary:
            end = boundary.start()
        raw = text[start:end]

        ir_idx = raw.find(_IR_MARKER)
        if ir_idx == -1:
            excerpt, has_ir, clause, supp = raw, False, "", ""
        else:
            excerpt = raw[:ir_idx]
            after = raw[ir_idx + len(_IR_MARKER) :]
            m = _CLAUSE.search(after)
            if m:
                clause = m.group(1)
                supp = after[m.end() :]
            else:
                clause, supp = "", after
            has_ir = True

        blocks.append(
            CommentBlock(
                commenter_number=int(a.group(1)),
                comment_excerpt=strip_running_headers(excerpt).strip(),
                has_individual_response=has_ir,
                raw_topic_clause=strip_running_headers(clause).strip(),
                individual_response_supplemental=strip_running_headers(supp).strip(),
            )
        )
    return blocks
