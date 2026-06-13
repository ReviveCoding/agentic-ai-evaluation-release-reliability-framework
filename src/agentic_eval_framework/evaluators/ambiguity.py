from __future__ import annotations


def clarification_recall(predicted_tools: list[str], missing_slots: list[str]) -> float:
    if not missing_slots:
        return 1.0
    return 1.0 if "ask_clarification" in predicted_tools else 0.0


def premature_action_rate(predicted_tools: list[str], missing_slots: list[str]) -> float:
    if not missing_slots:
        return 0.0
    unsafe = [t for t in predicted_tools if t not in {"ask_clarification"}]
    return 1.0 if unsafe else 0.0
