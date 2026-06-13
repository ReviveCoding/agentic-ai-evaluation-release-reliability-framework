from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RunStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    REVIEW = "REVIEW"
    BLOCKED = "BLOCKED"


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any]
    observation: dict[str, Any]
    latency_ms: float


@dataclass
class StepTrace:
    step_id: str
    predicted_tool: str
    tool_call: ToolCall | None = None
    evaluator_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class RunTrace:
    run_id: str
    scenario_id: str
    status: RunStatus = RunStatus.PENDING
    steps: list[StepTrace] = field(default_factory=list)
    release_decision: str | None = None
    failure_type: str | None = None
    latency_ms: float = 0.0
