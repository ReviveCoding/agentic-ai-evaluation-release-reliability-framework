from __future__ import annotations


def safety_check(query: str, risk_flags: list[str] | None = None) -> dict:
    return {
        "tool_name": "safety_check",
        "evidence_type": "safety",
        "evidence_id": "safety_review",
        "summary": f"Routed request to review due to risk flags: {risk_flags or []}",
        "grounded": True,
    }
