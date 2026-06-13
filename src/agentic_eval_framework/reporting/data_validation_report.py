from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_eval_framework.data.validate_dataset import validate_tool_policy_three_way


def export_data_validation_report(
    train_path: str | Path = "data/processed/tool_policy_train.jsonl",
    calibration_path: str | Path = "data/processed/tool_policy_calibration.jsonl",
    eval_path: str | Path = "data/processed/tool_policy_eval.jsonl",
    out_path: str | Path = "reports/data_validation_report.md",
) -> dict[str, Any]:
    result = validate_tool_policy_three_way(
        train_path, calibration_path, eval_path, "outputs/data_validation.json"
    )
    lines = [
        "# Data Validation Report",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- Train rows: {result['n_train']}",
        f"- Calibration rows: {result['n_calibration']}",
        f"- Final evaluation rows: {result['n_eval']}",
        f"- Reasons: {result['reasons'] or ['none']}",
        f"- Unseen calibration labels: {result['unseen_calibration_labels'] or ['none']}",
        f"- Unseen eval labels: {result['unseen_eval_labels'] or ['none']}",
        f"- Train/calibration ID overlap: {result['train_calibration_id_overlap'] or ['none']}",
        f"- Train/eval ID overlap: {result['train_eval_id_overlap'] or ['none']}",
        f"- Calibration/eval ID overlap: {result['calibration_eval_id_overlap'] or ['none']}",
        "",
        "## Label distribution",
        "",
        "### Train",
    ]
    lines.extend(f"- `{k}`: {v}" for k, v in result["train_label_counts"].items())
    lines += ["", "### Calibration"]
    lines.extend(f"- `{k}`: {v}" for k, v in result["calibration_label_counts"].items())
    lines += ["", "### Final evaluation"]
    lines.extend(f"- `{k}`: {v}" for k, v in result["eval_label_counts"].items())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
