from agentic_eval_framework.engine.release_gate import decide_release


def test_release_pass():
    decision, failure = decide_release({"tool_accuracy": 1, "trajectory_success": 1, "groundedness": 1, "clarification_recall": 1, "safety_block_rate": 1, "p95_latency_ms": 10, "trace_completeness": 1})
    assert decision == "PASS"
    assert failure is None


def test_release_block_safety():
    decision, failure = decide_release({"tool_accuracy": 1, "trajectory_success": 1, "groundedness": 1, "clarification_recall": 1, "safety_block_rate": 0, "p95_latency_ms": 10, "trace_completeness": 1})
    assert decision == "BLOCK"
    assert failure == "safety_failure"
