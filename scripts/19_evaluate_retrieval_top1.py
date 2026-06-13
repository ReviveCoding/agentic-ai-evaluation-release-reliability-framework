from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agentic_eval_framework.retrieval.evaluate_retrieval import evaluate_retrieval


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recency/tool-filtered retrieval and the lightweight reranker.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    outputs = Path(args.outputs_dir)
    outputs.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, scenarios in (
        ("id", "golden_trajectories.jsonl"),
        ("ood_dev", "golden_trajectories_ood_dev.jsonl"),
        ("ood_test", "golden_trajectories_ood_test.jsonl"),
    ):
        path = processed / scenarios
        if not path.exists():
            continue
        result[name] = evaluate_retrieval(
            scenarios_path=path,
            docs_path=processed / "service_docs.jsonl",
            out_path=outputs / f"retrieval_top1_{name}.json",
            top_k=args.top_k,
            query_mode="recency_weighted",
        )
    (outputs / "retrieval_top1_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
