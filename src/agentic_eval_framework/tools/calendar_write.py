from __future__ import annotations


def calendar_write(query: str, service: str = "") -> dict:
    """Stage a calendar write without committing an external side effect.

    The framework deliberately models write actions separately from read-only
    lookup. The returned draft requires review/confirmation before execution.
    """
    return {
        "tool_name": "calendar_write",
        "evidence_type": "calendar_action_draft",
        "evidence_id": f"draft::{service or 'calendar'}",
        "summary": f"Prepared a calendar action draft for: {query}",
        "grounded": True,
        "requires_confirmation": True,
        "side_effect_committed": False,
    }
