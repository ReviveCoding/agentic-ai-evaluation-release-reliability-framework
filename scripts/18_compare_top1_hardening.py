from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def pass_rate(payload: dict[str, Any], *prefix: str) -> float | None:
    aggregate = nested(payload, *prefix)
    if not isinstance(aggregate, dict):
        return None
    total = int(aggregate.get("num_runs", 0) or 0)
    passed = int((aggregate.get("release_decisions") or {}).get("PASS", 0) or 0)
    return passed / total if total else None


def change(after: Any, before: Any) -> float | None:
    return float(after) - float(before) if isinstance(after, (int, float)) and isinstance(before, (int, float)) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", default="outputs/top1_hardening_baseline")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--out-json", default="outputs/top1_hardening_comparison.json")
    parser.add_argument("--out-report", default="reports/final_retrieval_top1_hardening.md")
    args = parser.parse_args()

    base = Path(args.baseline_dir)
    out = Path(args.outputs_dir)
    before_id = read(base / "evaluation_results.json")
    after_id = read(out / "evaluation_results.json")
    before_ood = read(base / "ood_summary.json")
    after_ood = read(out / "ood_summary.json")
    before_replay = read(base / "replay_failures.json")
    after_replay = read(out / "replay_failures.json")

    metrics = [
        "retrieval_hit_at_k",
        "retrieval_strict_hit_at_k",
        "retrieval_top1_accuracy",
        "retrieval_strict_top1_accuracy",
        "retrieval_reciprocal_rank",
        "retrieval_strict_reciprocal_rank",
        "groundedness",
        "trajectory_success",
        "tool_accuracy",
    ]

    result: dict[str, Any] = {"id": {}, "ood_dev": {}, "ood_test": {}}
    for metric in metrics:
        before = nested(before_id, "aggregate", metric)
        after = nested(after_id, "aggregate", metric)
        result["id"][metric] = {"before": before, "after": after, "delta": change(after, before)}
        for split in ("ood_dev", "ood_test"):
            before = nested(before_ood, split, "agentic_aggregate", metric)
            after = nested(after_ood, split, "agentic_aggregate", metric)
            result[split][metric] = {"before": before, "after": after, "delta": change(after, before)}

    result["id"]["release_pass_rate"] = {
        "before": pass_rate(before_id, "aggregate"),
        "after": pass_rate(after_id, "aggregate"),
    }
    result["id"]["release_pass_rate"]["delta"] = change(
        result["id"]["release_pass_rate"]["after"],
        result["id"]["release_pass_rate"]["before"],
    )
    for split in ("ood_dev", "ood_test"):
        result[split]["release_pass_rate"] = {
            "before": pass_rate(before_ood, split, "agentic_aggregate"),
            "after": pass_rate(after_ood, split, "agentic_aggregate"),
        }
        result[split]["release_pass_rate"]["delta"] = change(
            result[split]["release_pass_rate"]["after"],
            result[split]["release_pass_rate"]["before"],
        )

    result["replay"] = {
        "before": before_replay.get("replay_verification_rate"),
        "after": after_replay.get("replay_verification_rate"),
        "delta": change(after_replay.get("replay_verification_rate"), before_replay.get("replay_verification_rate")),
    }

    id_top1_delta = nested(result, "id", "retrieval_top1_accuracy", "delta") or 0.0
    id_pass_delta = nested(result, "id", "release_pass_rate", "delta") or 0.0
    hit_delta = nested(result, "id", "retrieval_hit_at_k", "delta") or 0.0
    replay_after = result["replay"]["after"] or 0.0
    guards = {
        "compatible_hit_at_k_not_regressed": hit_delta >= -0.02,
        "replay_preserved": replay_after >= 0.99,
        "top1_meaningful_gain": id_top1_delta >= 0.02,
        "pass_rate_gain": id_pass_delta >= 0.01,
    }
    result["guards"] = guards
    result["final_status"] = (
        "PASS_MEANINGFUL_GAIN"
        if all(guards.values())
        else "PASS_TOP1_ONLY" if guards["compatible_hit_at_k_not_regressed"] and guards["replay_preserved"] and guards["top1_meaningful_gain"]
        else "MINOR_OR_NO_GAIN"
    )

    Path(args.out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Final Retrieval Top-1 Hardening Report",
        "",
        f"- Status: **{result['final_status']}**",
        f"- Replay verification after: {result['replay']['after']}",
        "",
    ]
    for split in ("id", "ood_dev", "ood_test"):
        lines += [
            f"## {split}",
            "",
            "| Metric | Before | After | Delta |",
            "|---|---:|---:|---:|",
        ]
        for metric, item in result[split].items():
            lines.append(f"| {metric} | {item.get('before')} | {item.get('after')} | {item.get('delta')} |")
        lines.append("")
    lines += ["## Guards", ""] + [f"- {key}: {value}" for key, value in guards.items()]
    Path(args.out_report).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
