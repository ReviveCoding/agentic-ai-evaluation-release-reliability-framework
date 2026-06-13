from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class ScenarioRequest(BaseModel):
    scenario_id: str = Field(default_factory=lambda: f"api_{uuid4().hex[:10]}")
    user_request: str = Field(min_length=1)
    service: str = "Hotels_1"
    intent: str = "SearchHotel"
    known_slots: dict[str, Any] = Field(default_factory=dict)
    missing_slots: list[str] = Field(default_factory=list)
    required_slots: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)
    must_not_tools: list[str] = Field(default_factory=list)
    expected_tools: list[str] | None = None
    release_gate_expected: Literal["PASS", "REVIEW", "BLOCK"] | None = None


class RunResponse(BaseModel):
    run_id: str
    scenario_id: str
    status: str
    release_decision: str
    failure_type: str | None = None
    predicted_tools: list[str]
    expected_tools: list[str]
    scores: dict[str, Any]
    model_fingerprint: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    model_backend: str
    model_fingerprint: str
    persistence_enabled: bool


class ReadyResponse(BaseModel):
    status: str
    database_integrity: str
    trace_payload_integrity: dict[str, int]
    model_loaded: bool
    retrieval_loaded: bool
    model_fingerprint: str
