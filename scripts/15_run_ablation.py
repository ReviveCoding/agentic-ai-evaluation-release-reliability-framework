from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agentic_eval_framework.models.ablation import (
    ABLATION_MODES,
    evaluate_linear_retrain_modes,
    evaluate_metadata_oracle,
    evaluate_permutation_sensitivity,
    evaluate_transformer_input_modes,
)
from agentic_eval_framework.utils.io import read_jsonl, write_json


def _predictor(model_dir: str, batch_size: int):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    id_to_label = {int(key): value for key, value in model.config.id2label.items()}

    def predict(texts: list[str]) -> list[str]:
        output: list[str] = []
        for start in range(0, len(texts), max(1, batch_size)):
            chunk = texts[start : start + max(1, batch_size)]
            batch = tokenizer(
                chunk, truncation=True, padding=True, max_length=256, return_tensors="pt"
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.inference_mode():
                predicted = model(**batch).logits.argmax(dim=-1)
            output.extend(id_to_label[int(index)] for index in predicted.detach().cpu().tolist())
        return output

    return predict


def export_markdown(result: dict, path: Path) -> None:
    transformer = result["transformer_inference_ablation"]["modes"]
    linear = result["linear_retrain_ablation"]
    lines = [
        "# ID Input Ablation Report",
        "",
        "This report separates model capability from structured-metadata shortcuts.",
        "",
        "## Same Transformer, masked inputs",
        "",
        "| Input mode | Accuracy | Macro-F1 | Drop vs full | Mean confidence |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in ABLATION_MODES:
        item = transformer[mode]
        lines.append(
            f"| {mode} | {item['accuracy']:.4f} | {item['macro_f1']:.4f} | "
            f"{item['macro_f1_drop_vs_full']:.4f} | {item['mean_confidence']:.4f} |"
        )
    lines += [
        "",
        "## Independently retrained linear baselines",
        "",
        "| Input mode | Accuracy | Macro-F1 | Drop vs full |",
        "|---|---:|---:|---:|",
    ]
    for mode in ABLATION_MODES:
        item = linear[mode]
        lines.append(
            f"| {mode} | {item['accuracy']:.4f} | {item['macro_f1']:.4f} | "
            f"{item['macro_f1_drop_vs_full']:.4f} |"
        )
    oracle = result["metadata_oracle"]
    lines += [
        "",
        "## Metadata lookup oracle",
        "",
        f"- Accuracy: {oracle['accuracy']:.4f}",
        f"- Macro-F1: {oracle['macro_f1']:.4f}",
        f"- Key coverage: {oracle['key_coverage']:.4f}",
        "",
        "## Permutation sensitivity",
        "",
        "| Permuted feature group | Accuracy | Macro-F1 | Macro-F1 drop |",
        "|---|---:|---:|---:|",
    ]
    for name, item in result["permutation_sensitivity"]["groups"].items():
        lines.append(
            f"| {name} | {item['accuracy']:.4f} | {item['macro_f1']:.4f} | {item['macro_f1_drop']:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model-dir", default="models/tool_policy")
    parser.add_argument("--out-path", default="outputs/ablation_summary.json")
    parser.add_argument("--report-path", default="reports/ablation_report.md")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    train_rows = read_jsonl(processed / "tool_policy_train.jsonl")
    eval_rows = read_jsonl(processed / "tool_policy_eval.jsonl")
    predictor = _predictor(args.model_dir, args.batch_size)

    result = {
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "transformer_inference_ablation": evaluate_transformer_input_modes(
            eval_rows, args.model_dir, batch_size=args.batch_size
        ),
        "linear_retrain_ablation": evaluate_linear_retrain_modes(train_rows, eval_rows),
        "metadata_oracle": evaluate_metadata_oracle(train_rows, eval_rows),
        "permutation_sensitivity": evaluate_permutation_sensitivity(eval_rows, predictor),
    }
    write_json(args.out_path, result)
    export_markdown(result, Path(args.report_path))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
