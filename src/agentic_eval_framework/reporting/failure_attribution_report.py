from __future__ import annotations

from collections import Counter
from pathlib import Path

from agentic_eval_framework.utils.io import ensure_dir, read_json


def export_failure_attribution_report(out_path: str | Path = "reports/failure_attribution_report.md") -> None:
    data = read_json("outputs/evaluation_results.json")
    runs = data.get("runs", [])
    failures = Counter(run.get("failure_type") or "none" for run in runs)
    text = "# Failure Attribution Report\n\n"
    text += "| Failure Type | Count |\n|---|---:|\n"
    for k, v in sorted(failures.items()):
        text += f"| {k} | {v} |\n"
    text += "\n## Example failures\n\n"
    for run in runs:
        if run.get("failure_type"):
            text += f"- `{run['run_id']}` scenario `{run['scenario_id']}`: {run['failure_type']}, predicted={run.get('predicted_tools')}, expected={run.get('expected_tools')}\n"
    p = Path(out_path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")
