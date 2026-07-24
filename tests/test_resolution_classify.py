import pytest

from stratification_scripts.resolution.classify import rule_class_from_action
from stratification_scripts.resolution.models import RuleClass


@pytest.mark.parametrize("action,expected", [
    ("Final rule.", RuleClass.FINAL),
    ("Final rule", RuleClass.FINAL),
    ("Final rule; technical amendment.", RuleClass.FINAL),
    ("Direct final rule.", RuleClass.DIRECT_FINAL),
    ("Direct final rule; request for comments.", RuleClass.DIRECT_FINAL),
    ("Interim final rule.", RuleClass.INTERIM_FINAL),
    ("Interim final action with comment period.", RuleClass.INTERIM_FINAL),
    ("Final rule with request for comments.", RuleClass.INTERIM_FINAL),
    ("Final rule; confirmation of effective date.", RuleClass.CONFIRMATION_OF_EFFECTIVE_DATE),
    ("Confirmation of effective date.", RuleClass.CONFIRMATION_OF_EFFECTIVE_DATE),
    ("Final rule; correction", RuleClass.CORRECTION),
    ("Correcting amendment.", RuleClass.CORRECTION),
    ("Notice of proposed rulemaking (NPRM).", RuleClass.PROPOSED),
    ("Proposed rule.", RuleClass.PROPOSED),
    ("Supplemental notice of proposed rulemaking.", RuleClass.PROPOSED),
    ("Notification of enforcement discretion.", RuleClass.OTHER),
    ("Notice of availability of fishery management plan.", RuleClass.OTHER),
    ("", RuleClass.OTHER),
    (None, RuleClass.OTHER),
])
def test_rule_class_from_action(action, expected):
    assert rule_class_from_action(action) is expected


def test_doc_type_only_backfills_when_action_is_empty():
    assert rule_class_from_action(None, doc_type="Rule") is RuleClass.FINAL
    assert rule_class_from_action(None, doc_type="Proposed Rule") is RuleClass.PROPOSED
    assert rule_class_from_action("Direct final rule.", doc_type="Rule") is RuleClass.DIRECT_FINAL


def test_deferred_response_variants_are_not_final():
    for action in ["Interim final action with comment period.",
                   "Final rule with request for comments."]:
        assert rule_class_from_action(action) is RuleClass.INTERIM_FINAL
