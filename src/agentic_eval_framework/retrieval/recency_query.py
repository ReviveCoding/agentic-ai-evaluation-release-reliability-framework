from __future__ import annotations

import re
from typing import Any

TURN_RE = re.compile(r"\[(USER|SYSTEM)\]\s*(.*?)(?=\s*\[(?:USER|SYSTEM)\]|$)", re.IGNORECASE)


def tagged_turns(context: str) -> list[tuple[str, str]]:
    return [(speaker.upper(), text.strip()) for speaker, text in TURN_RE.findall(str(context or "")) if text.strip()]


def current_utterance(payload: dict[str, Any]) -> str:
    return str(payload.get("user_request", payload.get("user_utterance", "")) or "").strip()


def recency_weighted_query(payload: dict[str, Any]) -> str:
    current = current_utterance(payload)
    turns = tagged_turns(str(payload.get("dialogue_context", "")))
    previous_user = next((text for speaker, text in reversed(turns) if speaker == "USER" and text != current), "")
    previous_system = next((text for speaker, text in reversed(turns) if speaker == "SYSTEM"), "")
    known_slots = payload.get("known_slots", {}) or {}
    slot_values = " ".join(
        str(value)
        for value in (known_slots.values() if isinstance(known_slots, dict) else [])
        if value not in (None, "")
    )
    parts = [current, current, current, current]
    if previous_user:
        parts.extend([previous_user, previous_user])
    if previous_system:
        parts.append(previous_system)
    if slot_values:
        parts.append(slot_values)
    return " ".join(part for part in parts if part).strip()
