from __future__ import annotations

from pathlib import Path

from agentic_eval_framework.utils.io import ensure_dir, read_jsonl


def export_dataset_card(
    out_path: str | Path = "reports/dataset_card.md",
    processed_dir: str | Path = "data/processed",
) -> None:
    processed = Path(processed_dir)
    train = read_jsonl(processed / "tool_policy_train.jsonl")
    calibration = read_jsonl(processed / "tool_policy_calibration.jsonl")
    eval_rows = read_jsonl(processed / "tool_policy_eval.jsonl")
    golden = read_jsonl(processed / "golden_trajectories.jsonl")
    labels = sorted({r.get("tool_label", "") for r in train + calibration + eval_rows})
    text = f"""# Dataset Card

## Source

Primary intended public source: Schema-Guided Dialogue, converted into agentic tool-policy labels and golden trajectories. This repository includes synthetic SGD-style sample data for offline smoke tests.

## Transformation

Dialogue turns are parsed into rows containing dialogue context, user utterance, service, intent, known slots, missing slots, risk flags, and a framework tool label. Train, calibration, and final evaluation splits are kept disjoint. The calibration split is used for confidence-threshold selection; final metrics and golden trajectories use only the held-out evaluation split.

## Sizes

- Train rows: {len(train)}
- Calibration rows: {len(calibration)}
- Final evaluation rows: {len(eval_rows)}
- Golden trajectories: {len(golden)}
- Tool labels: {', '.join(labels)}

## Claim boundary

This project uses public-data-compatible and synthetic sample inputs. It does not use Apple data, private user data, production Siri data, or real customer workflows.
"""
    p = Path(out_path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")
