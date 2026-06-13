from agentic_eval_framework.engine.release_gate import decide_release


def test_retry_gate_is_not_applied_when_no_retry_was_attempted():
    decision, failure = decide_release({
        "retry_recovery_applicable": 0.0,
        "retry_recovery_rate": 0.0,
    })
    assert decision == "PASS"
    assert failure is None


def test_retry_gate_applies_when_retry_was_attempted():
    decision, failure = decide_release({
        "retry_recovery_applicable": 1.0,
        "retry_recovery_rate": 0.0,
    })
    assert decision == "REVIEW"
    assert failure == "retry_recovery_failure"
