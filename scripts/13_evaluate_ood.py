from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agentic_eval_framework.engine.execution_engine import run_evaluation
from agentic_eval_framework.models.evaluate_tool_policy import evaluate_policies


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained policy on official SGD OOD dev/test splits.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model-dir", default="models/tool_policy")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    outputs = Path(args.outputs_dir)
    outputs.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {}
    for name in ("ood_dev", "ood_test"):
        classification = evaluate_policies(
            eval_path=processed / f"tool_policy_{name}.jsonl",
            model_dir=args.model_dir,
            out_path=outputs / f"tool_policy_{name}_classification.json",
        )
        agent = run_evaluation(
            scenarios_path=processed / f"golden_trajectories_{name}.jsonl",
            model_dir=args.model_dir,
            out_path=outputs / f"{name}_evaluation_results.json",
            traces_path=outputs / f"{name}_traces.jsonl",
            db_path=outputs / f"{name}_traces.sqlite",
            concurrency=args.concurrency,
            service_docs_path=processed / "service_docs.jsonl",
        )
        result[name] = {
            "classification": classification,
            "agentic_aggregate": agent["aggregate"],
        }
    (outputs / "ood_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
