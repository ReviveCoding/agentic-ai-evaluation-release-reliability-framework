from __future__ import annotations


def weather_lookup(query: str, service: str = "") -> dict:
    return {
        "tool_name": "weather_lookup",
        "evidence_type": "weather",
        "evidence_id": f"evidence::{service or 'weather'}",
        "summary": f"Retrieved weather information for: {query}",
        "grounded": True,
    }
