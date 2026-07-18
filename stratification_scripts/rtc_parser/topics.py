"""Topic structures: the 22 canonical topics, the per-topic Agency Topic
Discussion bodies, and resolution of Individual-Response cross-references to
canonical topics.

Canonical names are the verbatim `Agency Discussion on <Topic>` running-header
strings (copied exactly, including the en-dash in "Comments Related to Process –
Chemicals"). Cross-references in Individual Responses use short forms (PFAS,
DBPs) and sometimes name things outside the 22 topics (e.g. 1,4-Dioxane), so
resolution is best-effort and every mention is preserved with a resolved flag.
"""

from __future__ import annotations

import re

from stratification_scripts.rtc_parser.clean import strip_running_headers
from stratification_scripts.rtc_parser.models import TopicRef

CANONICAL_TOPICS: list[str] = [
    "General Comments",
    "Length of CCL 5",
    "Contaminant Groups",
    "Comments on Individual Chemical Contaminants",
    "Chemical Data/Data Sources",
    "Chemical Technical Support Documents",
    "Comments Related to Process – Chemicals",
    "Contaminants Not on the Draft CCL 5",
    "Suggestions to Improve the Process for Future CCLs",
    "Comment Outside the Scope of CCL",
    "Other Drinking Water Programs",
    "Other EPA Programs",
    "Per- and Polyfluoroalkyl substances (PFAS)",
    "Disinfection Byproducts (DBPs)",
    "EDCs and PPCPs",
    "Perchlorate",
    "Pesticides",
    "Cyanotoxins",
    "Draft CCL 5-Microbes",
    "Microbial Screening Process/Criteria",
    "Legionella pneumophila",
    "Mycobacterium",
]

_DISCUSSION_ANCHOR = "Agency Topic Discussion:"
_RUNNING_HEADER = re.compile(r"Agency Discussion on (.+)")
_COMMENT_EXCERPT = re.compile(r"Comment Excerpt from Commenter \d+")


def split_topic_discussions(text: str) -> dict[str, str]:
    """Map each canonical topic to its Agency Topic Discussion disposition text.

    For each `Agency Topic Discussion:` anchor, the topic is the nearest
    preceding `Agency Discussion on <Topic>` running header; the body runs from
    the anchor to the first following `Comment Excerpt from Commenter` or the next
    discussion anchor. Running headers inside the body are stripped.
    """
    out: dict[str, str] = {}
    anchors = [m.start() for m in re.finditer(re.escape(_DISCUSSION_ANCHOR), text)]
    for i, start in enumerate(anchors):
        header_matches = list(_RUNNING_HEADER.finditer(text, 0, start))
        if not header_matches:
            continue
        topic = header_matches[-1].group(1).strip()

        body_start = start + len(_DISCUSSION_ANCHOR)
        next_anchor = anchors[i + 1] if i + 1 < len(anchors) else len(text)
        excerpt = _COMMENT_EXCERPT.search(text, body_start, next_anchor)
        body_end = excerpt.start() if excerpt else next_anchor

        body = strip_running_headers(text[body_start:body_end]).strip()
        out[topic] = body
    return out
