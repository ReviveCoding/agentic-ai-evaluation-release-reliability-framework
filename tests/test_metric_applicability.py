from __future__ import annotations

from agentic_eval_framework.engine.execution_engine import aggregate_runs
from agentic_eval_framework.engine.release_gate import decide_release
from agentic_eval_framework.evaluators.retrieval_quality import retrieval_metrics


def test_no_retrieval_is_not_counted_as_perfect_retrieval():
    metrics = retrieval_metrics([{"tool_name": "safety_check"}])
    assert metrics["retrieval_applicable"] == 0.0
    assert metrics["retrieval_hit_at_k"] == 0.0


def test_release_gate_skips_non_applicable_metrics():
    decision, failure = decide_release({
        "retrieval_applicable": 0.0,
        "retrieval_hit_at_k": 0.0,
        "retrieval_reciprocal_rank": 0.0,
        "clarification_applicable": 0.0,
        "clarification_recall": 0.0,
        "safety_applicable": 0.0,
        "safety_block_rate": 0.0,
        "false_block_applicable": 0.0,
        "false_block_rate": 1.0,
    })
    assert decision == "PASS"
    assert failure is None


def test_aggregate_uses_conditional_metric_denominators():
    base = {"execution_stats": {}, "steps": [], "release_decision": "PASS"}
    runs = [
        {**base, "scores": {"retrieval_applicable": 1.0, "retrieval_hit_at_k": 0.0}},
        {**base, "scores": {"retrieval_applicable": 0.0, "retrieval_hit_at_k": 0.0}},
        {**base, "scores": {"retrieval_applicable": 1.0, "retrieval_hit_at_k": 1.0}},
    ]
    agg = aggregate_runs(runs)
    assert agg["retrieval_hit_at_k"] == 0.5
    assert agg["retrieval_hit_at_k_coverage"] == 2 / 3
