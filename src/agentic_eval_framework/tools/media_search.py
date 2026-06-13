from __future__ import annotations


def media_search(query: str, service: str = "") -> dict:
    return {
        "tool_name": "media_search",
        "evidence_type": "media",
        "evidence_id": f"evidence::{service or 'media'}",
        "summary": f"Retrieved media results for: {query}",
        "grounded": True,
    }
