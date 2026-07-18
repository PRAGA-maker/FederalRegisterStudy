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


# High-confidence expansions for short forms substring-matching would also catch;
# kept explicit to document intent. Everything else resolves by exact match or a
# unique substring, or stays unresolved.
_ALIASES = {
    "pfas": "Per- and Polyfluoroalkyl substances (PFAS)",
    "dbps": "Disinfection Byproducts (DBPs)",
}
_CANON_BY_LOWER = {c.lower(): c for c in CANONICAL_TOPICS}

# Split a cross-ref clause on commas that are NOT inside a number like "1,4"
# (so the chemical name "1,4-Dioxane" survives).
_COMMA_SEP = re.compile(r",(?!\d)")


def _resolve_one(mention: str) -> str | None:
    """Resolve one mention to a canonical topic, or None (never force-match)."""
    low = mention.strip().lower()
    if not low:
        return None
    if low in _CANON_BY_LOWER:
        return _CANON_BY_LOWER[low]
    if low in _ALIASES:
        return _ALIASES[low]
    subs = [c for c in CANONICAL_TOPICS if low in c.lower()]
    return subs[0] if len(subs) == 1 else None


def _split_mentions(clause: str) -> list[str]:
    """Tokenize a cross-ref clause into individual topic mentions.

    Commas separate (except inside numbers); a segment that does not itself
    resolve but contains " and " is split further, so `EDCs and PPCPs` (a real
    topic) stays whole while `General Comments and PFAS` splits into two.
    """
    norm = re.sub(r"\s+", " ", clause).strip()
    if not norm:
        return []
    mentions: list[str] = []
    for seg in _COMMA_SEP.split(norm):
        seg = seg.strip()
        if seg.lower().startswith("and "):  # Oxford "…, and X"
            seg = seg[4:].strip()
        if not seg:
            continue
        if _resolve_one(seg) is None and " and " in seg:
            mentions.extend(s.strip() for s in seg.split(" and ") if s.strip())
        else:
            mentions.append(seg)
    return mentions


def resolve_refs(raw_clause: str) -> list[TopicRef]:
    """Resolve an Individual-Response cross-ref clause to TopicRefs.

    Every mention is preserved with its verbatim `raw`; `canonical`/`resolved`
    record whether it mapped to one of the 22 topics. Nothing is dropped.
    """
    refs: list[TopicRef] = []
    for mention in _split_mentions(raw_clause):
        canonical = _resolve_one(mention)
        refs.append(TopicRef(raw=mention, canonical=canonical, resolved=canonical is not None))
    return refs
