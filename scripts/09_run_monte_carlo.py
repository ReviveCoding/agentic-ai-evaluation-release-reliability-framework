from __future__ import annotations

from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import argparse
import json

from agentic_eval_framework.simulation.monte_carlo import MonteCarloConfig, run_monte_carlo_validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Monte Carlo end-to-end validation.")
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--train-dialogues", type=int, default=900)
    parser.add_argument("--dev-dialogues", type=int, default=300, help="Calibration/model-selection dialogues.")
    parser.add_argument(
        "--test-dialogues", type=int, default=None,
        help="Held-out final evaluation dialogues; defaults to --dev-dialogues.",
    )
    parser.add_argument("--replications", type=int, default=12)
    parser.add_argument("--scenarios-per-replication", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--backend", choices=["sklearn", "transformer"], default="sklearn")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    config = MonteCarloConfig(
        seed=args.seed,
        n_train_dialogues=args.train_dialogues,
        n_dev_dialogues=args.dev_dialogues,
        n_test_dialogues=args.test_dialogues,
        replications=args.replications,
        scenarios_per_replication=args.scenarios_per_replication,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        training_backend=args.backend,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    result = run_monte_carlo_validation(config)
    compact = {
        "total_scenarios": result["total_scenarios"],
        "total_runs": result["total_runs"],
        "training_macro_f1": result["training"].get("macro_f1"),
        "all_sanity_checks_passed": result["all_sanity_checks_passed"],
        "variants": {k: v["aggregate"] for k, v in result["summary"].items()},
    }
    print(json.dumps(compact, indent=2))
