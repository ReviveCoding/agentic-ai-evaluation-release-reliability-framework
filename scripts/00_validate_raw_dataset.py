from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agentic_eval_framework.data.dataset_contract import validate_sgd_raw_dataset
from agentic_eval_framework.utils.io import ensure_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate an SGD raw dataset directory before running the pipeline.")
    parser.add_argument("--raw-dir", default="data/raw/sgd")
    parser.add_argument("--skip-json-inspection", action="store_true")
    parser.add_argument("--report", default="reports/raw_dataset_validation.json")
    args = parser.parse_args()

    result = validate_sgd_raw_dataset(args.raw_dir, inspect_json=not args.skip_json_inspection)
    report_path = Path(args.report)
    ensure_dir(report_path.parent)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
