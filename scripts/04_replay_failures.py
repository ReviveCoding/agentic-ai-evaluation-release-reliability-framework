from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agentic_eval_framework.engine.replay import replay_failed_runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="outputs/traces.sqlite")
    parser.add_argument("--out-path", default="outputs/replay_failures.json")
    parser.add_argument("--model-dir", default="models/tool_policy")
    parser.add_argument("--service-docs", default="data/processed/service_docs.jsonl")
    parser.add_argument("--release-gates", default="configs/release_gates.yaml")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--report-only", action="store_true", help="Do not re-execute stored scenarios")
    args = parser.parse_args()
    result = replay_failed_runs(
        db_path=args.db_path,
        out_path=args.out_path,
        reexecute=not args.report_only,
        model_dir=args.model_dir,
        service_docs_path=args.service_docs,
        release_gates_path=args.release_gates,
        max_retries=args.max_retries,
    )
    print(json.dumps({
        "num_failed_or_review_runs": result["num_failed_or_review_runs"],
        "num_replay_verified": result["num_replay_verified"],
        "replay_verification_rate": result["replay_verification_rate"],
        "mode_summary": result.get("mode_summary", {}),
        "mismatch_counts": result.get("mismatch_counts", {}),
    }, indent=2))
