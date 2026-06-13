from __future__ import annotations


def search_places(query: str, service: str = "") -> dict:
    return {
        "tool_name": "search_places",
        "evidence_type": "places",
        "evidence_id": f"evidence::{service or 'places'}",
        "summary": f"Retrieved place/service options for query: {query}",
        "grounded": True,
    }
