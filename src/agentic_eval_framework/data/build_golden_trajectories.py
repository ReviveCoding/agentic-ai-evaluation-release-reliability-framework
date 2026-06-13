from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_eval_framework.utils.io import read_jsonl, write_jsonl


def expected_tools_for_row(row: dict[str, Any]) -> list[str]:
    tool = row["tool_label"]
    if tool in {"ask_clarification", "safety_check", "calendar_write"}:
        return [tool]
    return [tool, "final_answer"]


def build_golden_trajectories(
    dataset_path: str | Path = "data/processed/tool_policy_eval.jsonl",
    out_path: str | Path = "data/processed/golden_trajectories.jsonl",
) -> list[dict[str, Any]]:
    rows = read_jsonl(dataset_path)
    trajectories: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        tools = expected_tools_for_row(row)
        must_not = []
        if row["tool_label"] not in {"safety_check"}:
            must_not.append("safety_check")
        if row["tool_label"] != "ask_clarification" and row.get("missing_slots"):
            must_not.append("final_answer")
        trajectories.append({
            "scenario_id": f"S{idx+1:05d}",
            "source_dataset": "SGD-style",
            "source_example_id": row["example_id"],
            "user_request": row["user_utterance"],
            "dialogue_context": row.get("dialogue_context", ""),
            "service": row["service"],
            "intent": row["intent"],
            "known_slots": row.get("known_slots", {}),
            "missing_slots": row.get("missing_slots", []),
            "expected_tools": tools,
            "required_slots": row.get("required_slots", []),
            "must_not_tools": must_not,
            "expected_outcome": (
                "review_required" if row["tool_label"] == "safety_check"
                else "confirmation_required" if row["tool_label"] == "calendar_write"
                else "clarification" if row["tool_label"] == "ask_clarification"
                else "answer_with_options"
            ),
            "release_gate_expected": "REVIEW" if row["tool_label"] in {"safety_check", "calendar_write"} else "PASS",
            "risk_flags": row.get("risk_flags", []),
        })
    write_jsonl(out_path, trajectories)
    return trajectories
