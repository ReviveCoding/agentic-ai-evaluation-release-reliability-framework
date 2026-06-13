from agentic_eval_framework.models.rule_policy import RulePolicy


def test_rule_policy_missing_slot():
    row = {"service": "Calendar_1", "intent": "CreateEvent", "missing_slots": ["date"]}
    assert RulePolicy().predict_one(row) == "ask_clarification"


def test_rule_policy_sensitive():
    row = {"service": "Bank_1", "intent": "TransferMoney", "missing_slots": []}
    assert RulePolicy().predict_one(row) == "safety_check"
