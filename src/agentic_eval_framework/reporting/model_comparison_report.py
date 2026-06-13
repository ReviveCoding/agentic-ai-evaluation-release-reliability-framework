from __future__ import annotations

from pathlib import Path

from agentic_eval_framework.utils.io import ensure_dir, read_json


def export_model_comparison_report(out_path: str | Path = "reports/model_comparison_report.md") -> None:
    tool_eval = read_json("outputs/tool_policy_eval.json") if Path("outputs/tool_policy_eval.json").exists() else {}
    train_metrics = read_json("models/tool_policy/metrics.json") if Path("models/tool_policy/metrics.json").exists() else {}
    num_examples = tool_eval.get("num_examples", 0)
    text = "# Model Comparison Report\n\n"
    if num_examples and num_examples < 100:
        text += "> Smoke-test warning: this report was generated on a small sample dataset. Use full/subset public SGD results for resume metrics.\n\n"
    text += f"Evaluation examples: **{num_examples}**\n\n"
    text += "| Variant | Backend | Macro-F1 | Notes |\n|---|---|---:|---|\n"
    rule = tool_eval.get("rule_baseline", {})
    trained = tool_eval.get("trained_policy", {})
    text += f"| Rule baseline | deterministic | {rule.get('macro_f1', 0.0):.4f} | service/intent/slot rules |\n"
    text += f"| Trained tool-policy | {trained.get('backend', train_metrics.get('backend', 'unknown'))} | {trained.get('macro_f1', train_metrics.get('macro_f1', 0.0)):.4f} | learned from processed tool-policy labels |\n"
    p = Path(out_path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")
