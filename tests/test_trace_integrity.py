from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agentic_eval_framework.engine.execution_engine import AgenticExecutionEngine
from agentic_eval_framework.storage.sqlite_store import SQLiteTraceStore
from agentic_eval_framework.utils.trace_integrity import verify_trace_fingerprint


def _scenario() -> dict:
    return {
        "scenario_id": "trace_1",
        "user_request": "What meetings do I have tomorrow?",
        "service": "Calendar_1",
        "intent": "GetEvents",
        "known_slots": {"date": "tomorrow"},
        "missing_slots": [],
        "required_slots": ["date"],
        "risk_flags": [],
        "must_not_tools": [],
        "expected_tools": ["calendar_lookup", "final_answer"],
        "release_gate_expected": "PASS",
    }


def test_trace_payload_tamper_detection(tmp_path: Path):
    db = tmp_path / "traces.sqlite"
    engine = AgenticExecutionEngine(
        model_dir=tmp_path / "missing_model",
        db_path=db,
        service_docs_path=tmp_path / "missing_docs.jsonl",
    )
    try:
        run = asyncio.run(engine.run_scenario(_scenario(), predicted_first="calendar_lookup", policy_route="rule", confidence_applicable=False))
        run_id = run["run_id"]
        assert verify_trace_fingerprint(run)
        assert engine.store.payload_integrity_summary() == {"checked": 1, "valid": 1, "invalid": 0, "missing": 0}
    finally:
        engine.close()

    store = SQLiteTraceStore(db)
    try:
        payload = store.get_run(run_id)
        assert payload is not None
        payload["release_decision"] = "BLOCK"
        store.conn.execute("UPDATE runs SET payload_json = ? WHERE run_id = ?", (json.dumps(payload), run_id))
        store.conn.commit()
        summary = store.payload_integrity_summary()
    finally:
        store.close()
    assert summary["invalid"] == 1
    assert summary["valid"] == 0
