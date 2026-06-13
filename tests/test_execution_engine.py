import asyncio

from agentic_eval_framework.engine.execution_engine import AgenticExecutionEngine


def test_execution_engine_rule_fallback(tmp_path):
    scenario = {
        "scenario_id": "S1",
        "user_request": "What meetings do I have tomorrow?",
        "service": "Calendar_1",
        "intent": "GetEvents",
        "known_slots": {"date": "tomorrow"},
        "missing_slots": [],
        "risk_flags": [],
        "expected_tools": ["calendar_lookup", "final_answer"],
    }
    engine = AgenticExecutionEngine(
        model_dir=tmp_path / "missing_model",
        db_path=tmp_path / "traces.sqlite",
        service_docs_path=tmp_path / "missing_service_docs.jsonl",
    )
    run = asyncio.run(engine.run_scenario(scenario))
    assert run["predicted_tools"] == ["calendar_lookup", "final_answer"]
    assert run["release_decision"] == "PASS"
