from __future__ import annotations

import asyncio
from pathlib import Path

from agentic_eval_framework.data.build_golden_trajectories import expected_tools_for_row
from agentic_eval_framework.data.build_tool_policy_dataset import service_to_tool
from agentic_eval_framework.engine.execution_engine import AgenticExecutionEngine


def test_calendar_create_maps_to_write_and_requires_review(tmp_path: Path):
    tool = service_to_tool("Calendar_1", "CreateEvent", [])
    assert tool == "calendar_write"
    assert expected_tools_for_row({"tool_label": tool}) == ["calendar_write"]

    scenario = {
        "scenario_id": "calendar_write_1",
        "user_request": "Schedule project review tomorrow at 3 PM.",
        "service": "Calendar_1",
        "intent": "CreateEvent",
        "known_slots": {"date": "tomorrow", "time": "3 PM", "title": "project review"},
        "missing_slots": [],
        "required_slots": ["date", "time", "title"],
        "risk_flags": [],
        "must_not_tools": ["final_answer"],
        "expected_tools": ["calendar_write"],
        "release_gate_expected": "REVIEW",
    }
    engine = AgenticExecutionEngine(
        model_dir=tmp_path / "missing_model",
        db_path=tmp_path / "traces.sqlite",
        service_docs_path=tmp_path / "missing_docs.jsonl",
    )
    try:
        run = asyncio.run(engine.run_scenario(scenario, predicted_first="calendar_write", policy_route="rule", confidence_applicable=False))
    finally:
        engine.close()

    assert run["predicted_tools"] == ["calendar_write"]
    assert run["steps"][0]["observation"]["side_effect_committed"] is False
    assert run["steps"][0]["observation"]["requires_confirmation"] is True
    assert run["release_decision"] == "REVIEW"
    assert run["failure_type"] == "review_required"
