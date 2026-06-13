from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.metrics import classification_report, f1_score

from agentic_eval_framework.models.rule_policy import RulePolicy
from agentic_eval_framework.models.transformer_tool_policy import ToolPolicyModel
from agentic_eval_framework.utils.io import read_jsonl, write_json


def evaluate_policies(eval_path: str | Path = "data/processed/tool_policy_eval.jsonl", model_dir: str | Path = "models/tool_policy", out_path: str | Path = "outputs/tool_policy_eval.json") -> dict[str, Any]:
    rows = read_jsonl(eval_path)
    y_true = [r["tool_label"] for r in rows]
    rule_preds = RulePolicy().predict(rows)
    model = ToolPolicyModel(model_dir)
    model_preds = model.predict(rows)
    labels = sorted(set(y_true) | set(rule_preds) | set(model_preds))
    result = {
        "num_examples": len(rows),
        "rule_baseline": {
            "macro_f1": float(f1_score(y_true, rule_preds, labels=labels, average="macro", zero_division=0)),
            "classification_report": classification_report(y_true, rule_preds, labels=labels, output_dict=True, zero_division=0),
        },
        "trained_policy": {
            "backend": model.backend,
            "macro_f1": float(f1_score(y_true, model_preds, labels=labels, average="macro", zero_division=0)),
            "classification_report": classification_report(y_true, model_preds, labels=labels, output_dict=True, zero_division=0),
        },
    }
    write_json(out_path, result)
    return result
