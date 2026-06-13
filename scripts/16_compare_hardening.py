from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def get_metric(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def delta(after: Any, before: Any) -> float | None:
    if isinstance(after, (int, float)) and isinstance(before, (int, float)):
        return float(after) - float(before)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", default="outputs/hardening_baseline")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--out-json", default="outputs/hardening_comparison.json")
    parser.add_argument("--out-report", default="reports/retrieval_replay_ablation_hardening.md")
    args = parser.parse_args()

    baseline = Path(args.baseline_dir)
    outputs = Path(args.outputs_dir)
    before_id = read_json(baseline / "evaluation_results.json")
    after_id = read_json(outputs / "evaluation_results.json")
    before_replay = read_json(baseline / "replay_failures.json")
    after_replay = read_json(outputs / "replay_failures.json")
    before_ood = read_json(baseline / "ood_summary.json")
    after_ood = read_json(outputs / "ood_summary.json")
    ablation = read_json(outputs / "ablation_summary.json")

    id_metrics = [
        "tool_accuracy",
        "retrieval_hit_at_k",
        "retrieval_top1_accuracy",
        "retrieval_reciprocal_rank",
        "groundedness",
        "trajectory_success",
    ]
    id_comparison: dict[str, Any] = {}
    for metric in id_metrics:
        before = get_metric(before_id, "aggregate", metric)
        after = get_metric(after_id, "aggregate", metric)
        id_comparison[metric] = {"before": before, "after": after, "delta": delta(after, before)}

    replay_comparison = {
        "before": before_replay.get("replay_verification_rate"),
        "after": after_replay.get("replay_verification_rate"),
        "delta": delta(after_replay.get("replay_verification_rate"), before_replay.get("replay_verification_rate")),
        "mode_summary": after_replay.get("mode_summary", {}),
        "mismatch_counts": after_replay.get("mismatch_counts", {}),
    }

    ood_comparison: dict[str, Any] = {}
    for split in ("ood_dev", "ood_test"):
        ood_comparison[split] = {}
        for metric in ("tool_accuracy", "retrieval_hit_at_k", "retrieval_top1_accuracy", "groundedness", "trajectory_success"):
            before = get_metric(before_ood, split, "agentic_aggregate", metric)
            after = get_metric(after_ood, split, "agentic_aggregate", metric)
            ood_comparison[split][metric] = {"before": before, "after": after, "delta": delta(after, before)}

    modes = get_metric(ablation, "transformer_inference_ablation", "modes") or {}
    linear_modes = ablation.get("linear_retrain_ablation", {})
    oracle = ablation.get("metadata_oracle", {})
    full_f1 = get_metric(modes, "full", "macro_f1")
    utterance_f1 = get_metric(modes, "utterance_context_only", "macro_f1")
    metadata_f1 = get_metric(modes, "metadata_names_only", "macro_f1")
    schema_f1 = get_metric(modes, "schema_descriptions_only", "macro_f1")

    interpretation: list[str] = []
    if isinstance(metadata_f1, (int, float)) and isinstance(full_f1, (int, float)) and metadata_f1 >= full_f1 - 0.02:
        interpretation.append("Metadata names alone nearly reproduce the full-model score, indicating a strong schema-to-tool shortcut.")
    if isinstance(utterance_f1, (int, float)) and isinstance(full_f1, (int, float)) and utterance_f1 < full_f1 - 0.15:
        interpretation.append("Utterance/context-only performance drops materially, so the perfect ID result should not be described as pure language understanding.")
    if isinstance(schema_f1, (int, float)) and isinstance(full_f1, (int, float)) and schema_f1 >= full_f1 - 0.05:
        interpretation.append("Natural-language schema descriptions are sufficient for most of the ID task.")
    if float(oracle.get("accuracy", 0.0) or 0.0) >= 0.99 and float(oracle.get("key_coverage", 0.0) or 0.0) >= 0.95:
        interpretation.append("A train-derived metadata lookup oracle solves almost all ID examples, confirming that the tool labels are largely deterministic functions of structured SGD annotations.")
    if not interpretation:
        interpretation.append("No single input group reproduces the full score; the result appears to depend on multiple information sources.")

    result = {
        "id_comparison": id_comparison,
        "replay_comparison": replay_comparison,
        "ood_comparison": ood_comparison,
        "ablation": {
            "transformer_modes": modes,
            "linear_modes": linear_modes,
            "metadata_oracle": oracle,
            "interpretation": interpretation,
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# Retrieval, Replay, and ID Ablation Hardening Report",
        "",
        "## ID before/after",
        "",
        "| Metric | Before | After | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric, item in id_comparison.items():
        before = item["before"]
        after = item["after"]
        change = item["delta"]
        lines.append(
            f"| {metric} | {before if before is not None else 'N/A'} | {after if after is not None else 'N/A'} | {change if change is not None else 'N/A'} |"
        )
    lines += [
        "",
        "## Replay",
        "",
        f"- Before: {replay_comparison['before']}",
        f"- After: {replay_comparison['after']}",
        f"- Mode summary: `{json.dumps(replay_comparison['mode_summary'], sort_keys=True)}`",
        f"- Mismatch counts: `{json.dumps(replay_comparison['mismatch_counts'], sort_keys=True)}`",
        "",
        "## ID 100% interpretation",
        "",
    ]
    lines.extend(f"- {item}" for item in interpretation)
    lines += ["", "See `reports/ablation_report.md` for the complete ablation table.", ""]
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
