from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agentic_eval_framework.data.build_golden_trajectories import build_golden_trajectories
from agentic_eval_framework.data.dataset_contract import validate_sgd_raw_dataset
from agentic_eval_framework.data.sgd_stratified import (
    audit_raw_mapping,
    build_service_docs_from_training_rows,
    build_stratified_sgd_datasets,
)
from agentic_eval_framework.reporting.data_validation_report import export_data_validation_report
from agentic_eval_framework.reporting.dataset_card import export_dataset_card


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build stratified internal SGD splits and separate OOD dev/test benchmarks."
    )
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--max-dialogues", type=int, default=250)
    parser.add_argument("--max-ood-dialogues", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-rows-per-label", type=int, default=3)
    args = parser.parse_args()

    validation = validate_sgd_raw_dataset(args.raw_dir)
    if validation["status"] != "PASS":
        raise SystemExit(f"Raw SGD dataset validation failed: {validation['errors']}")

    mapping = audit_raw_mapping(args.raw_dir)
    if mapping["status"] != "PASS":
        raise SystemExit(f"SGD mapping audit failed: {mapping['failures']}")

    processed = Path(args.processed_dir)
    manifest = build_stratified_sgd_datasets(
        raw_dir=args.raw_dir,
        processed_dir=processed,
        max_train_dialogues=args.max_dialogues,
        max_ood_dialogues=args.max_ood_dialogues,
        seed=args.seed,
        minimum_rows_per_label=args.minimum_rows_per_label,
    )
    from agentic_eval_framework.utils.io import read_jsonl
    build_service_docs_from_training_rows(
        args.raw_dir,
        read_jsonl(processed / "tool_policy_train.jsonl"),
        processed / "service_docs.jsonl",
    )
    build_golden_trajectories(
        processed / "tool_policy_eval.jsonl",
        processed / "golden_trajectories.jsonl",
    )
    for name in ("ood_dev", "ood_test"):
        build_golden_trajectories(
            processed / f"tool_policy_{name}.jsonl",
            processed / f"golden_trajectories_{name}.jsonl",
        )
    export_dataset_card(processed_dir=processed)
    validation = export_data_validation_report(
        processed / "tool_policy_train.jsonl",
        processed / "tool_policy_calibration.jsonl",
        processed / "tool_policy_eval.jsonl",
    )
    if validation["status"] != "PASS":
        raise SystemExit(f"Processed dataset validation failed: {validation['reasons']}")
    print(json.dumps({"status": "PASS", "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
