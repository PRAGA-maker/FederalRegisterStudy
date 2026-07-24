"""Document resolution layer.

Answers "where could a response to this comment live?" — returns every candidate
document found across five declared channels, with provenance, rule classification,
and a three-valued status. Never decides whether the agency actually responded.

Standalone: not imported by the pipeline cli.
"""

from .models import (  # noqa: F401
    AbsenceReason, AgendaStatus, CandidateDocument, Channel, CommentRef,
    Relevance, ResolutionResult, ResponseEvidence, RuleClass, Status,
)
