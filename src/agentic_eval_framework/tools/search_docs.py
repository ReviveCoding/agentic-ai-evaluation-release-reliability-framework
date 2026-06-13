from __future__ import annotations


def search_docs(query: str, service: str = "") -> dict:
    return {
        "tool_name": "search_docs",
        "evidence_type": "documents",
        "evidence_id": f"evidence::{service or 'documents'}",
        "summary": f"Retrieved document evidence for query: {query}",
        "grounded": True,
    }
