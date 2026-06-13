from __future__ import annotations

from pathlib import Path

from agentic_eval_framework.evaluators.latency import latency_summary
from agentic_eval_framework.utils.io import ensure_dir, read_json


def export_latency_report(out_path: str | Path = "reports/latency_report.md") -> None:
    data = read_json("outputs/evaluation_results.json")
    runs = data.get("runs", [])
    latencies = [step.get("latency_ms", 0.0) for run in runs for step in run.get("steps", [])]
    summary = latency_summary(latencies)
    text = "# Latency Report\n\n"
    text += "| Metric | Value ms |\n|---|---:|\n"
    for k, v in summary.items():
        text += f"| {k} | {v:.4f} |\n"
    p = Path(out_path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")
