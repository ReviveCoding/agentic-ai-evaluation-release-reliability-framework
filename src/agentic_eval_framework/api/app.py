from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request

from agentic_eval_framework import __version__
from agentic_eval_framework.api.schemas import HealthResponse, ReadyResponse, RunResponse, ScenarioRequest
from agentic_eval_framework.data.build_golden_trajectories import expected_tools_for_row
from agentic_eval_framework.data.build_tool_policy_dataset import service_to_tool
from agentic_eval_framework.engine.execution_engine import AgenticExecutionEngine
from agentic_eval_framework.tools.registry import TOOL_REGISTRY


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    engine = AgenticExecutionEngine(
        model_dir=os.getenv("AGENTIC_MODEL_DIR", "models/tool_policy"),
        db_path=os.getenv("AGENTIC_TRACE_DB", "outputs/api_traces.sqlite"),
        release_gates_path=os.getenv("AGENTIC_RELEASE_GATES", "configs/release_gates.yaml"),
        service_docs_path=os.getenv("AGENTIC_SERVICE_DOCS", "data/processed/service_docs.jsonl"),
    )
    app.state.engine = engine
    try:
        yield
    finally:
        engine.close()


app = FastAPI(title="Agentic Evaluation Framework", version=__version__, lifespan=lifespan)


def _engine(request: Request) -> AgenticExecutionEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Execution engine is not initialized")
    return engine


def _validate_tools(tools: list[str]) -> None:
    unknown = sorted(set(tools) - set(TOOL_REGISTRY))
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown tool names: {unknown}")


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    engine = _engine(request)
    return HealthResponse(
        status="ok",
        version=__version__,
        model_backend=engine.policy.backend,
        model_fingerprint=engine.policy.model_fingerprint,
        persistence_enabled=engine.persist_runs,
    )


@app.get("/ready", response_model=ReadyResponse)
def ready(request: Request) -> ReadyResponse:
    engine = _engine(request)
    integrity = engine.store.integrity_check()
    trace_integrity = engine.store.payload_integrity_summary(limit=1000)
    model_loaded = engine.policy.backend == "rule" or engine.policy.model_dir.exists()
    payload_ok = trace_integrity["invalid"] == 0 and trace_integrity["missing"] == 0
    ready_status = "ready" if integrity == "ok" and model_loaded and payload_ok else "not_ready"
    return ReadyResponse(
        status=ready_status,
        database_integrity=integrity,
        trace_payload_integrity=trace_integrity,
        model_loaded=model_loaded,
        retrieval_loaded=engine.retriever is not None,
        model_fingerprint=engine.policy.model_fingerprint,
    )


@app.post("/runs", response_model=RunResponse)
async def create_run(req: ScenarioRequest, request: Request) -> RunResponse:
    inferred_first = service_to_tool(req.service, req.intent, req.missing_slots)
    expected_tools = req.expected_tools or expected_tools_for_row({"tool_label": inferred_first})
    _validate_tools(expected_tools)
    _validate_tools(req.must_not_tools)
    gate_expected = req.release_gate_expected or ("REVIEW" if inferred_first in {"safety_check", "calendar_write"} else "PASS")
    scenario = {
        "scenario_id": req.scenario_id,
        "user_request": req.user_request,
        "service": req.service,
        "intent": req.intent,
        "known_slots": req.known_slots,
        "missing_slots": req.missing_slots,
        "required_slots": req.required_slots,
        "risk_flags": req.risk_flags,
        "must_not_tools": req.must_not_tools,
        "expected_tools": expected_tools,
        "oracle_source": "request" if req.expected_tools is not None else "inferred_schema_mapping",
        "release_gate_expected": gate_expected,
    }
    run = await _engine(request).run_scenario(scenario, variant="api")
    return RunResponse(
        run_id=run["run_id"],
        scenario_id=run["scenario_id"],
        status=run["status"],
        release_decision=run["release_decision"],
        failure_type=run.get("failure_type"),
        predicted_tools=run["predicted_tools"],
        expected_tools=run["expected_tools"],
        scores=run["scores"],
        model_fingerprint=run.get("model_fingerprint"),
    )


@app.get("/runs/{run_id}", response_model=RunResponse)
def get_run(run_id: str, request: Request) -> RunResponse:
    run = _engine(request).store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunResponse(
        run_id=run["run_id"],
        scenario_id=run["scenario_id"],
        status=run["status"],
        release_decision=run["release_decision"],
        failure_type=run.get("failure_type"),
        predicted_tools=run["predicted_tools"],
        expected_tools=run["expected_tools"],
        scores=run["scores"],
        model_fingerprint=run.get("model_fingerprint"),
    )
