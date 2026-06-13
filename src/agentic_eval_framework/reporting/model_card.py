from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentic_eval_framework.models.model_manifest import load_model_manifest


def export_model_card(
    model_dir: str | Path = "models/tool_policy",
    out_path: str | Path = "reports/model_card.md",
) -> dict[str, Any]:
    model_dir = Path(model_dir)
    manifest = load_model_manifest(model_dir)
    metrics_path = model_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    lines = [
        "# Tool-Policy Model Card",
        "",
        f"- Backend: `{manifest.get('backend', metrics.get('backend', 'unknown'))}`",
        f"- Model fingerprint: `{manifest.get('model_fingerprint', 'unavailable')}`",
        f"- Artifact SHA-256: `{manifest.get('artifact_sha256', 'unavailable')}`",
        f"- Train data SHA-256: `{manifest.get('train_data', {}).get('sha256', 'unavailable')}`",
        f"- Calibration data SHA-256: `{(manifest.get('calibration_data') or {}).get('sha256', 'unavailable')}`",
        f"- Eval data SHA-256: `{manifest.get('eval_data', {}).get('sha256', 'unavailable')}`",
        f"- Recommended minimum confidence: `{manifest.get('recommended_min_confidence')}`",
        f"- Threshold-selection split: `{manifest.get('training_config', {}).get('threshold_selection_split', 'unknown')}`",
        "",
        "## Evaluation metrics",
        "",
    ]
    for key in ("macro_f1", "ece", "multiclass_brier", "log_loss", "eval_macro_f1", "eval_ece", "eval_multiclass_brier", "eval_log_loss"):
        if key in metrics:
            lines.append(f"- `{key}`: `{metrics[key]}`")
    lines += [
        "",
        "## Intended use",
        "",
        "The model predicts the next tool/action for public or synthetic task-oriented dialogue scenarios inside this local evaluation framework.",
        "",
        "## Limitations",
        "",
        "The model is not a production assistant, does not execute real-world actions, and should not be treated as an autonomous safety decision-maker.",
    ]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"manifest": manifest, "metrics": metrics}
