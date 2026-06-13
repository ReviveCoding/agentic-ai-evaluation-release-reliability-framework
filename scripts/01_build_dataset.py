from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agentic_eval_framework.data.build_golden_trajectories import build_golden_trajectories
from agentic_eval_framework.data.build_service_docs import build_service_docs
from agentic_eval_framework.data.build_tool_policy_dataset import build_tool_policy_dataset
from agentic_eval_framework.data.dataset_contract import validate_sgd_raw_dataset
from agentic_eval_framework.data.download_sgd import download_sgd
from agentic_eval_framework.data.parse_sgd import create_sample_raw
from agentic_eval_framework.reporting.dataset_card import export_dataset_card
from agentic_eval_framework.reporting.data_validation_report import export_data_validation_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-sample", action="store_true", help="Create and use included synthetic SGD-style sample data.")
    parser.add_argument("--download-sgd", action="store_true", help="Download and normalize the public SGD repository before building.")
    parser.add_argument("--raw-dir", default="data/raw/sgd")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--max-dialogues", type=int, default=None, help="Optional per-split dialogue cap for quick public-data validation.")
    parser.add_argument("--download-cache-dir", default="data/raw/sgd_download")
    args = parser.parse_args()

    if args.use_sample and args.download_sgd:
        parser.error("Choose either --use-sample or --download-sgd, not both.")
    if args.use_sample:
        create_sample_raw(args.raw_dir)
    elif args.download_sgd:
        download_sgd(args.raw_dir, args.download_cache_dir)

    raw_validation = validate_sgd_raw_dataset(args.raw_dir)
    if raw_validation["status"] != "PASS":
        raise SystemExit(f"Raw SGD dataset validation failed: {raw_validation['errors']}")

    processed = Path(args.processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    build_tool_policy_dataset(
        args.raw_dir,
        split="train",
        out_path=processed / "tool_policy_train.jsonl",
        max_dialogues=args.max_dialogues,
    )
    build_tool_policy_dataset(
        args.raw_dir,
        split="dev",
        out_path=processed / "tool_policy_calibration.jsonl",
        max_dialogues=args.max_dialogues,
    )
    build_tool_policy_dataset(
        args.raw_dir,
        split="test",
        out_path=processed / "tool_policy_eval.jsonl",
        max_dialogues=args.max_dialogues,
    )
    build_service_docs(
        args.raw_dir,
        split=("train", "dev", "test"),
        out_path=processed / "service_docs.jsonl",
    )
    build_golden_trajectories(
        processed / "tool_policy_eval.jsonl",
        processed / "golden_trajectories.jsonl",
    )
    export_dataset_card(processed_dir=processed)
    validation = export_data_validation_report(
        train_path=processed / "tool_policy_train.jsonl",
        calibration_path=processed / "tool_policy_calibration.jsonl",
        eval_path=processed / "tool_policy_eval.jsonl",
    )
    if validation["status"] != "PASS":
        raise SystemExit(f"Processed dataset validation failed: {validation['reasons']}")
    print(f"Built processed dataset and golden trajectories in {processed}.")
