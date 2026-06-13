from __future__ import annotations


def tool_accuracy(predicted_tools: list[str], expected_tools: list[str]) -> float:
    if not expected_tools:
        return 1.0
    if not predicted_tools:
        return 0.0
    return 1.0 if predicted_tools[0] == expected_tools[0] else 0.0
