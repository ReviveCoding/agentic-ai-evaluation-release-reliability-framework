from __future__ import annotations


def reconstruct_lineage(run: dict) -> list[dict]:
    lineage = []
    for step in run.get("steps", []):
        obs = step.get("observation", {})
        lineage.append({
            "run_id": run.get("run_id"),
            "scenario_id": run.get("scenario_id"),
            "step_id": step.get("step_id"),
            "predicted_tool": step.get("predicted_tool"),
            "evidence_id": obs.get("evidence_id"),
            "release_decision": run.get("release_decision"),
            "failure_type": run.get("failure_type"),
        })
    return lineage
