from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agentic_eval_framework.data.build_golden_trajectories import build_golden_trajectories
from agentic_eval_framework.data.build_service_docs import build_service_docs
from agentic_eval_framework.data.build_tool_policy_dataset import build_tool_policy_dataset
from agentic_eval_framework.data.parse_sgd import create_sample_raw
from agentic_eval_framework.engine.execution_engine import run_evaluation
from agentic_eval_framework.engine.replay import replay_failed_runs
from agentic_eval_framework.models.train_tool_policy import train_tool_policy
from agentic_eval_framework.reporting.dataset_card import export_dataset_card
from agentic_eval_framework.reporting.export_reports import export_all_reports


def run_smoke_pipeline() -> dict:
    create_sample_raw("data/raw/sgd")
    train_rows = build_tool_policy_dataset("data/raw/sgd", "train", "data/processed/tool_policy_train.jsonl")
    calibration_rows = build_tool_policy_dataset(
        "data/raw/sgd", "dev", "data/processed/tool_policy_calibration.jsonl"
    )
    eval_rows = build_tool_policy_dataset("data/raw/sgd", "test", "data/processed/tool_policy_eval.jsonl")
    build_service_docs("data/raw/sgd", ("train", "dev", "test"), "data/processed/service_docs.jsonl")
    golden = build_golden_trajectories("data/processed/tool_policy_eval.jsonl", "data/processed/golden_trajectories.jsonl")
    export_dataset_card()
    train_metrics = train_tool_policy(backend="sklearn")
    evaluation = run_evaluation()
    replay = replay_failed_runs()
    export_all_reports()
    return {
        "train_rows": len(train_rows),
        "calibration_rows": len(calibration_rows),
        "eval_rows": len(eval_rows),
        "golden_scenarios": len(golden),
        "train_backend": train_metrics.get("backend"),
        "train_macro_f1": train_metrics.get("macro_f1"),
        "evaluation": evaluation.get("aggregate", {}),
        "replay": replay.get("num_failed_or_review_runs", 0),
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke_pipeline(), indent=2))
