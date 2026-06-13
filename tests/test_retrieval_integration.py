from __future__ import annotations

import asyncio

from agentic_eval_framework.data.build_service_docs import build_service_docs
from agentic_eval_framework.data.parse_sgd import create_sample_raw
from agentic_eval_framework.engine.execution_engine import AgenticExecutionEngine


def _hotel_scenario() -> dict:
    return {
        "scenario_id": "retrieval_hotel",
        "user_request": "Find a hotel in Cambridge with free wifi",
        "service": "Hotels_1",
        "intent": "SearchHotel",
        "known_slots": {"location": "Cambridge"},
        "missing_slots": [],
        "required_slots": ["location"],
        "risk_flags": [],
        "must_not_tools": ["safety_check"],
        "expected_tools": ["search_places", "final_answer"],
        "release_gate_expected": "PASS",
    }


def test_retrieval_is_part_of_agent_execution(tmp_path):
    raw = tmp_path / "raw"
    create_sample_raw(raw)
    docs = tmp_path / "service_docs.jsonl"
    build_service_docs(raw, ("train", "dev"), docs)
    engine = AgenticExecutionEngine(
        model_dir=tmp_path / "missing_model",
        db_path=tmp_path / "traces.sqlite",
        service_docs_path=docs,
    )
    try:
        run = asyncio.run(engine.run_scenario(_hotel_scenario(), predicted_first="search_places"))
    finally:
        engine.close()
    first = run["steps"][0]["observation"]
    assert first["retrieved_doc_ids"]
    assert first["retrieval_target_doc_id"] == "intent::Hotels_1::SearchHotel"
    assert first["evidence_id"] in {"intent::Hotels_1::SearchHotel", "service::Hotels_1"}
    assert run["scores"]["retrieval_hit_at_k"] == 1.0
    assert run["scores"]["retrieval_top1_accuracy"] == 1.0


def test_wrong_retrieval_is_not_grounded(tmp_path):
    raw = tmp_path / "raw"
    create_sample_raw(raw)
    docs = tmp_path / "service_docs.jsonl"
    build_service_docs(raw, ("train", "dev"), docs)
    scenario = _hotel_scenario()
    scenario.update({
        "user_request": "please check the forecast",
        "observed_service": "Weather_1",
        "observed_intent": "GetWeather",
    })
    engine = AgenticExecutionEngine(
        model_dir=tmp_path / "missing_model",
        db_path=tmp_path / "traces.sqlite",
        service_docs_path=docs,
    )
    try:
        run = asyncio.run(engine.run_scenario(scenario, predicted_first="search_places"))
    finally:
        engine.close()
    assert run["scores"]["retrieval_top1_accuracy"] == 0.0
    assert run["scores"]["groundedness"] == 0.0
    assert run["release_decision"] == "REVIEW"
