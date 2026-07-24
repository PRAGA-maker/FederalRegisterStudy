"""Rule classification from the Federal Register `action` field."""

from __future__ import annotations

import re
from typing import Optional

from .models import RuleClass

_PATTERNS = [
    (re.compile(r"confirmation\s+of\s+effective\s+date", re.I),
     RuleClass.CONFIRMATION_OF_EFFECTIVE_DATE),
    (re.compile(r"\bcorrect(ion|ing|ions)\b", re.I), RuleClass.CORRECTION),
    (re.compile(r"direct\s+final", re.I), RuleClass.DIRECT_FINAL),
    (re.compile(r"interim\s+final", re.I), RuleClass.INTERIM_FINAL),
    (re.compile(r"final\s+rule.*(request\s+for\s+comment|comment\s+period)", re.I),
     RuleClass.INTERIM_FINAL),
    (re.compile(r"\b(proposed\s+rule|nprm|proposed\s+rulemaking)\b", re.I),
     RuleClass.PROPOSED),
    (re.compile(r"final\s+(rule|action)", re.I), RuleClass.FINAL),
]

_DOC_TYPE_FALLBACK = {
    "Rule": RuleClass.FINAL,
    "Proposed Rule": RuleClass.PROPOSED,
}


def rule_class_from_action(
    action: Optional[str], doc_type: Optional[str] = None
) -> RuleClass:
    """Classify an FR document from its `action` string."""
    text = (action or "").strip()
    if not text:
        return _DOC_TYPE_FALLBACK.get((doc_type or "").strip(), RuleClass.OTHER)
    for pattern, rule_class in _PATTERNS:
        if pattern.search(text):
            return rule_class
    return RuleClass.OTHER
