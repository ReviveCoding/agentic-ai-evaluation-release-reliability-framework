from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agentic_eval_framework.engine.execution_engine import run_evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--append", action="store_true", help="Append runs to the existing trace database instead of resetting outputs.")
    parser.add_argument("--release-gates", default="configs/release_gates.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model-dir", default="models/tool_policy")
    parser.add_argument("--outputs-dir", default="outputs")
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    outputs = Path(args.outputs_dir)
    result = run_evaluation(
        scenarios_path=processed / "golden_trajectories.jsonl",
        model_dir=args.model_dir,
        out_path=outputs / "evaluation_results.json",
        traces_path=outputs / "traces.jsonl",
        db_path=outputs / "traces.sqlite",
        concurrency=args.concurrency,
        append=args.append,
        release_gates_path=args.release_gates,
        service_docs_path=processed / "service_docs.jsonl",
    )
    print(json.dumps(result["aggregate"], indent=2))
