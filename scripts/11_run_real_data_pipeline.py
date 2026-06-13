from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agentic_eval_framework.data.build_golden_trajectories import build_golden_trajectories
from agentic_eval_framework.data.build_service_docs import build_service_docs
from agentic_eval_framework.data.build_tool_policy_dataset import build_tool_policy_dataset
from agentic_eval_framework.data.dataset_contract import validate_sgd_raw_dataset
from agentic_eval_framework.engine.execution_engine import run_evaluation
from agentic_eval_framework.engine.replay import replay_failed_runs
from agentic_eval_framework.models.train_tool_policy import train_tool_policy
from agentic_eval_framework.reporting.data_validation_report import export_data_validation_report
from agentic_eval_framework.reporting.dataset_card import export_dataset_card
from agentic_eval_framework.reporting.export_reports import export_all_reports
from agentic_eval_framework.reporting.model_card import export_model_card


def run_pipeline(args: argparse.Namespace) -> dict:
    raw = Path(args.raw_dir).expanduser().resolve()
    processed = ROOT_DIR / "data" / "processed"
    model_dir = ROOT_DIR / "models" / "tool_policy"
    outputs = ROOT_DIR / "outputs"

    raw_validation = validate_sgd_raw_dataset(raw, inspect_json=not args.fast_validate)
    if raw_validation["status"] != "PASS":
        raise ValueError(f"Raw dataset contract failed: {raw_validation['errors']}")

    processed.mkdir(parents=True, exist_ok=True)
    train_rows = build_tool_policy_dataset(raw, "train", processed / "tool_policy_train.jsonl", args.max_dialogues)
    calibration_rows = build_tool_policy_dataset(raw, "dev", processed / "tool_policy_calibration.jsonl", args.max_dialogues)
    eval_rows = build_tool_policy_dataset(raw, "test", processed / "tool_policy_eval.jsonl", args.max_dialogues)
    docs = build_service_docs(raw, ("train", "dev", "test"), processed / "service_docs.jsonl")
    golden = build_golden_trajectories(processed / "tool_policy_eval.jsonl", processed / "golden_trajectories.jsonl")
    export_dataset_card()
    processed_validation = export_data_validation_report(
        processed / "tool_policy_train.jsonl",
        processed / "tool_policy_calibration.jsonl",
        processed / "tool_policy_eval.jsonl",
    )
    if processed_validation["status"] != "PASS":
        raise ValueError(f"Processed dataset validation failed: {processed_validation['reasons']}")

    metrics = train_tool_policy(
        train_path=processed / "tool_policy_train.jsonl",
        calibration_path=processed / "tool_policy_calibration.jsonl",
        eval_path=processed / "tool_policy_eval.jsonl",
        out_dir=model_dir,
        backend=args.backend,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    export_model_card(model_dir=model_dir)
    evaluation = run_evaluation(
        scenarios_path=processed / "golden_trajectories.jsonl",
        model_dir=model_dir,
        out_path=outputs / "evaluation_results.json",
        traces_path=outputs / "traces.jsonl",
        db_path=outputs / "traces.sqlite",
        concurrency=args.concurrency,
        service_docs_path=processed / "service_docs.jsonl",
    )
    replay = replay_failed_runs(
        db_path=outputs / "traces.sqlite",
        out_path=outputs / "replay_failures.json",
        model_dir=model_dir,
        service_docs_path=processed / "service_docs.jsonl",
    )
    export_all_reports()
    return {
        "status": "PASS",
        "raw_dir": str(raw),
        "backend": args.backend,
        "train_rows": len(train_rows),
        "calibration_rows": len(calibration_rows),
        "eval_rows": len(eval_rows),
        "service_docs": len(docs),
        "golden_scenarios": len(golden),
        "model_metrics": metrics,
        "evaluation": evaluation["aggregate"],
        "replay_verification_rate": replay.get("replay_verification_rate"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the complete framework from an existing SGD raw dataset directory.")
    parser.add_argument("--raw-dir", required=True, help="Directory containing train/dev/test SGD split folders.")
    parser.add_argument("--backend", choices=["sklearn", "transformer"], default="sklearn")
    parser.add_argument("--model-name", default="distilbert-base-uncased", help="Hugging Face model ID or local model directory.")
    parser.add_argument("--max-dialogues", type=int, default=None, help="Optional per-split cap for a quick validation run.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--fast-validate", action="store_true", help="Check layout without loading every JSON file.")
    args = parser.parse_args()
    print(json.dumps(run_pipeline(args), indent=2))
