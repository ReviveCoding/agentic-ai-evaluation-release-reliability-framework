from agentic_eval_framework.engine.release_gate import decide_release


def test_release_gate_uses_max_step_latency_name():
    scores = {
        "tool_accuracy": 1.0,
        "trajectory_success": 1.0,
        "groundedness": 1.0,
        "clarification_recall": 1.0,
        "safety_block_rate": 1.0,
        "trace_completeness": 1.0,
        "model_confidence": 1.0,
        "max_step_latency_ms": 1001.0,
    }
    decision, failure = decide_release(scores, {"max_step_latency_ms": 1000.0})
    assert decision == "REVIEW"
    assert failure == "latency_failure"
