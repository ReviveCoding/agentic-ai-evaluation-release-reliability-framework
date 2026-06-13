from pathlib import Path

from agentic_eval_framework.engine.release_gate import load_release_thresholds, decide_release


def test_release_gate_config_override(tmp_path):
    cfg = tmp_path / "release_gates.yaml"
    cfg.write_text("thresholds:\n  min_tool_accuracy: 0.99\n", encoding="utf-8")
    thresholds = load_release_thresholds(cfg)
    assert thresholds["min_tool_accuracy"] == 0.99
    decision, failure = decide_release({
        "tool_accuracy": 0.95,
        "trajectory_success": 1,
        "groundedness": 1,
        "clarification_recall": 1,
        "safety_block_rate": 1,
        "p95_latency_ms": 10,
        "trace_completeness": 1,
    }, thresholds)
    assert decision == "REVIEW"
    assert failure == "policy_model_failure"
