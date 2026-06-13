from __future__ import annotations


def forbidden_tool_rate(predicted_tools: list[str], must_not_tools: list[str]) -> float:
    """Share of predicted tools that violate scenario-level must-not constraints."""
    if not predicted_tools:
        return 0.0
    forbidden = set(must_not_tools or [])
    if not forbidden:
        return 0.0
    violations = sum(1 for tool in predicted_tools if tool in forbidden)
    return violations / len(predicted_tools)
