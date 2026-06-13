import asyncio

from agentic_eval_framework.engine.execution_engine import AgenticExecutionEngine


def _calendar_scenario():
    return {
        "scenario_id": "retry_case",
        "user_request": "What meetings do I have tomorrow?",
        "service": "Calendar_1",
        "intent": "GetEvents",
        "known_slots": {"date": "tomorrow"},
        "missing_slots": [],
        "required_slots": ["date"],
        "risk_flags": [],
        "expected_tools": ["calendar_lookup", "final_answer"],
        "must_not_tools": ["safety_check"],
    }


def test_retry_recovers_transient_failure(tmp_path, monkeypatch):
    engine = AgenticExecutionEngine(
        model_dir=tmp_path / "missing_model",
        db_path=tmp_path / "traces.sqlite",
        service_docs_path=tmp_path / "missing_service_docs.jsonl",
        max_retries=1,
    )

    def fake_fault(scenario, tool_name, attempt):
        return "transient_error" if tool_name == "calendar_lookup" and attempt == 0 else None

    monkeypatch.setattr(engine, "_fault_type", fake_fault)
    scenario = _calendar_scenario()
    scenario["fault_profile"] = {"seed": 1, "latency_base_ms": 1, "latency_jitter_ms": 1}
    run = asyncio.run(engine.run_scenario(scenario, predicted_first="calendar_lookup"))
    engine.close()

    first = run["steps"][0]["execution"]
    assert first["retry_count"] == 1
    assert first["recovered_after_retry"] is True
    assert run["scores"]["execution_error_rate"] == 0.0
    assert run["scores"]["groundedness"] == 1.0


def test_wrong_tool_evidence_is_not_grounded(tmp_path):
    engine = AgenticExecutionEngine(
        model_dir=tmp_path / "missing",
        db_path=tmp_path / "traces.sqlite",
        service_docs_path=tmp_path / "missing_service_docs.jsonl",
    )
    run = asyncio.run(engine.run_scenario(_calendar_scenario(), predicted_first="search_places"))
    engine.close()
    assert run["scores"]["tool_accuracy"] == 0.0
    assert run["scores"]["groundedness"] == 0.0
    assert run["release_decision"] == "REVIEW"
