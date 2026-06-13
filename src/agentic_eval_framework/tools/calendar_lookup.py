from __future__ import annotations


def calendar_lookup(query: str, service: str = "") -> dict:
    return {
        "tool_name": "calendar_lookup",
        "evidence_type": "calendar",
        "evidence_id": f"evidence::{service or 'calendar'}",
        "summary": f"Retrieved calendar context for: {query}",
        "grounded": True,
    }
