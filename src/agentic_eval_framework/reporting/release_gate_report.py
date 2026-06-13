from __future__ import annotations

from pathlib import Path

from agentic_eval_framework.utils.io import ensure_dir, read_json


def export_release_gate_report(out_path: str | Path = "reports/release_gate_report.md") -> None:
    data = read_json("outputs/evaluation_results.json")
    agg = data.get("aggregate", {})
    decisions = agg.get("release_decisions", {})
    text = "# Release Gate Report\n\n"
    text += "| Decision | Count |\n|---|---:|\n"
    for k in ["PASS", "REVIEW", "BLOCK"]:
        text += f"| {k} | {decisions.get(k, 0)} |\n"
    text += "\n## Aggregate scores\n\n"
    text += "| Metric | Value |\n|---|---:|\n"
    for k, v in sorted(agg.items()):
        if k != "release_decisions":
            text += f"| {k} | {v:.4f} |\n" if isinstance(v, float) else f"| {k} | {v} |\n"
    p = Path(out_path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")
