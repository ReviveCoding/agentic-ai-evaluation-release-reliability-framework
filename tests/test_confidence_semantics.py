from __future__ import annotations

from agentic_eval_framework.engine.execution_engine import aggregate_runs
from agentic_eval_framework.engine.release_gate import decide_release


def test_confidence_gate_only_applies_to_probabilistic_route():
    decision, failure = decide_release({"model_confidence": 0.0, "confidence_applicable": 0.0})
    assert decision == "PASS"
    assert failure is None

    decision, failure = decide_release({"model_confidence": 0.0, "confidence_applicable": 1.0})
    assert decision == "REVIEW"
    assert failure == "low_confidence"


def test_aggregate_reports_confidence_coverage():
    runs = [
        {
            "scores": {"confidence_applicable": 0.0, "model_confidence": 0.0},
            "execution_stats": {},
            "steps": [],
            "release_decision": "PASS",
        },
        {
            "scores": {"confidence_applicable": 1.0, "model_confidence": 0.8},
            "execution_stats": {},
            "steps": [],
            "release_decision": "PASS",
        },
    ]
    agg = aggregate_runs(runs)
    assert agg["confidence_coverage"] == 0.5
    assert agg["model_confidence"] == 0.8
