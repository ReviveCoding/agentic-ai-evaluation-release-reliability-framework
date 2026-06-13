from __future__ import annotations


def final_answer(query: str, evidence: dict | None = None) -> dict:
    evidence = evidence or {}
    evidence_id = evidence.get("evidence_id", "none")
    compatible = bool(evidence.get("evidence_compatible", evidence.get("grounded", False))) and evidence_id not in {"none", "error", "corrupt"}
    return {
        "tool_name": "final_answer",
        "source_tool": evidence.get("tool_name"),
        "evidence_type": evidence.get("evidence_type", "unknown"),
        "evidence_id": evidence_id,
        "summary": f"Final answer grounded in {evidence_id} for request: {query}",
        "grounded": compatible,
        "evidence_compatible": compatible,
    }
