"""Graded response evidence, derived from the extract — never a gate."""

from __future__ import annotations

from typing import Optional

from ..makeup.fr_response_extractor import DENSITY_KW, ResponseExtract
from .models import ResponseEvidence

STRONG_DENSITY_PER_1K = 2.0


def density_per_1k(text: str) -> float:
    """Comment-discussion keyword hits per 1,000 characters of grounded text."""
    if not text:
        return 0.0
    return len(DENSITY_KW.findall(text)) / (len(text) / 1000.0)


def response_evidence_from_extract(
    extract: Optional[ResponseExtract],
) -> ResponseEvidence:
    """Grade evidence that a candidate's preamble discusses comments."""
    if extract is None or not extract.grounded_text:
        return ResponseEvidence.NONE
    if extract.found_response_hd:
        return ResponseEvidence.STRONG
    if density_per_1k(extract.grounded_text) >= STRONG_DENSITY_PER_1K:
        return ResponseEvidence.STRONG
    return ResponseEvidence.WEAK
