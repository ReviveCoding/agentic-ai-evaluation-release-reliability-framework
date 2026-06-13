from __future__ import annotations


def ask_clarification(query: str, missing_slots: list[str] | None = None) -> dict:
    missing = missing_slots or []
    return {
        "tool_name": "ask_clarification",
        "evidence_type": "clarification",
        "evidence_id": "clarification",
        "summary": f"Asked clarification for missing slots: {missing}",
        "grounded": True,
    }
