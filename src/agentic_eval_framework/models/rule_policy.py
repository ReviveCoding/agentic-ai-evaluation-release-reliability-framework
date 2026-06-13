from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_eval_framework.data.build_tool_policy_dataset import service_to_tool


@dataclass
class RulePolicy:
    name: str = "rule_baseline"

    def predict_one(self, row: dict[str, Any]) -> str:
        return service_to_tool(row.get("service", ""), row.get("intent", ""), row.get("missing_slots", []))

    def predict(self, rows: list[dict[str, Any]]) -> list[str]:
        return [self.predict_one(r) for r in rows]
