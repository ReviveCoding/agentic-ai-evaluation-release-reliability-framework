from __future__ import annotations


def exact_trajectory_success(predicted_tools: list[str], expected_tools: list[str]) -> float:
    return 1.0 if predicted_tools == expected_tools else 0.0


def partial_trajectory_success(predicted_tools: list[str], expected_tools: list[str]) -> float:
    if not expected_tools:
        return 1.0
    matches = sum(1 for p, e in zip(predicted_tools, expected_tools) if p == e)
    return matches / len(expected_tools)
