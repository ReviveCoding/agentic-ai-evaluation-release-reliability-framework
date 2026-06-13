from __future__ import annotations

from pathlib import Path

from agentic_eval_framework.storage.lineage import reconstruct_lineage
from agentic_eval_framework.utils.io import ensure_dir, read_json


def export_trace_lineage_report(out_path: str | Path = "reports/trace_lineage_report.md") -> None:
    data = read_json("outputs/evaluation_results.json")
    runs = data.get("runs", [])
    text = "# Trace Lineage Report\n\n"
    text += "Each row maps release decision back to scenario, step, tool call, and evidence.\n\n"
    text += "| Run | Scenario | Step | Tool | Evidence | Decision | Failure |\n|---|---|---|---|---|---|---|\n"
    for run in runs[:50]:
        for item in reconstruct_lineage(run):
            text += f"| {item['run_id']} | {item['scenario_id']} | {item['step_id']} | {item['predicted_tool']} | {item['evidence_id']} | {item['release_decision']} | {item['failure_type']} |\n"
    p = Path(out_path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")
