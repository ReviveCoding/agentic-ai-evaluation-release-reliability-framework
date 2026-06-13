from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    command = [sys.executable, *args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run stratified SGD build, model training, ID evaluation, and official dev/test OOD evaluation."
    )
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--backend", choices=["sklearn", "transformer"], default="sklearn")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-dialogues", type=int, default=250)
    parser.add_argument("--max-ood-dialogues", type=int, default=250)
    parser.add_argument("--minimum-rows-per-label", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    run(
        "scripts/12_build_stratified_sgd.py",
        "--raw-dir", args.raw_dir,
        "--max-dialogues", str(args.max_dialogues),
        "--max-ood-dialogues", str(args.max_ood_dialogues),
        "--minimum-rows-per-label", str(args.minimum_rows_per_label),
    )
    train_args = [
        "scripts/02_train_model.py",
        "--backend", args.backend,
        "--model-name", args.model_name,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
    ]
    if args.gradient_checkpointing:
        train_args.append("--gradient-checkpointing")
    run(*train_args)
    run("scripts/17_train_retrieval_reranker.py")
    run("scripts/03_run_evaluation.py")
    run("scripts/04_replay_failures.py")
    run("scripts/05_export_reports.py")
    run("scripts/13_evaluate_ood.py", "--concurrency", str(args.concurrency))


if __name__ == "__main__":
    main()
