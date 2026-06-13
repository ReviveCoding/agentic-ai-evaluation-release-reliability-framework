from __future__ import annotations


def groundedness_score(observations: list[dict]) -> float:
    """Return the share of observations that are both grounded and tool-compatible."""
    if not observations:
        return 0.0
    grounded = [
        bool(obs.get("grounded"))
        and bool(obs.get("evidence_compatible", True))
        and obs.get("evidence_id") not in {None, "none", "error", "corrupt"}
        for obs in observations
    ]
    return sum(grounded) / len(grounded)
