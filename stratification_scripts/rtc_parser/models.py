"""Record dataclasses for the RTC crosswalk."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Commenter:
    """One Exhibit 2 row: a commenter number joined to a regulations.gov Document ID."""

    number: int
    document_id: str
    first_name: str
    last_name: str
    organization: str


@dataclass(frozen=True)
class TopicRef:
    """A single topic mention from an Individual Response cross-reference.

    `raw` is the verbatim mention; `canonical` is the resolved 22-topic name or
    None; `resolved` says whether resolution succeeded. Unresolvable mentions are
    kept (resolved=False, canonical=None), never dropped and never force-matched.
    """

    raw: str
    canonical: str | None
    resolved: bool


@dataclass
class CommentRecord:
    """One comment excerpt's crosswalk row: Document ID -> topic(s) -> disposition."""

    commenter_number: int
    document_id: str | None
    first_name: str
    last_name: str
    organization: str
    comment_excerpt: str
    has_individual_response: bool
    topic_refs: list[TopicRef] = field(default_factory=list)
    individual_response_supplemental: str = ""
    topic_discussions: dict[str, str] = field(default_factory=dict)
