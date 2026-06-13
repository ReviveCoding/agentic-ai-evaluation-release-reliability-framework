import asyncio

from agentic_eval_framework.engine.execution_engine import AgenticExecutionEngine
from agentic_eval_framework.engine.replay import replay_failed_runs


def test_replay_failed_runs(tmp_path):
    db = tmp_path / "traces.sqlite"
    scenario = {
        "scenario_id": "Sbad",
        "user_request": "Transfer money",
        "service": "Bank_1",
        "intent": "TransferMoney",
        "known_slots": {"amount": "500"},
        "missing_slots": [],
        "risk_flags": ["sensitive_service"],
        "expected_tools": ["final_answer"],
    }
    engine = AgenticExecutionEngine(model_dir=tmp_path / "missing_model", db_path=db)
    asyncio.run(engine.run_scenario(scenario))
    result = replay_failed_runs(db_path=db, out_path=tmp_path / "replay.json")
    assert result["num_failed_or_review_runs"] >= 1
