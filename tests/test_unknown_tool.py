from __future__ import annotations

import asyncio

from agentic_eval_framework.engine.execution_engine import AgenticExecutionEngine


def test_unknown_tool_is_explicit_execution_failure(tmp_path):
    scenario = {
        "scenario_id": "unknown_tool",
        "user_request": "Do the task",
        "service": "Documents_1",
        "intent": "SearchDocument",
        "missing_slots": [],
        "risk_flags": [],
        "must_not_tools": [],
        "expected_tools": ["search_docs", "final_answer"],
        "release_gate_expected": "PASS",
    }
    engine = AgenticExecutionEngine(
        model_dir=tmp_path / "missing_model",
        db_path=tmp_path / "traces.sqlite",
    )
    try:
        run = asyncio.run(engine.run_scenario(scenario, predicted_first="not_a_registered_tool"))
    finally:
        engine.close()
    assert run["steps"][0]["observation"]["unknown_tool"] is True
    assert run["scores"]["execution_error_rate"] > 0
    assert run["release_decision"] == "REVIEW"
    assert run["failure_type"] == "execution_failure"
