from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agentic_eval_framework.models.train_tool_policy import train_tool_policy
from agentic_eval_framework.reporting.model_card import export_model_card


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["sklearn", "transformer"], default="sklearn")
    parser.add_argument("--model-name", default="distilbert-base-uncased", help="Hugging Face model ID or local pretrained-model directory.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-dir", default="models/tool_policy")
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    metrics = train_tool_policy(
        train_path=processed / "tool_policy_train.jsonl",
        calibration_path=processed / "tool_policy_calibration.jsonl",
        eval_path=processed / "tool_policy_eval.jsonl",
        out_dir=args.out_dir,
        backend=args.backend,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    export_model_card(model_dir=args.out_dir)
    print(json.dumps(metrics, indent=2))
