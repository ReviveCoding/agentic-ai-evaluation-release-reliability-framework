from __future__ import annotations

import inspect

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, log_loss

from agentic_eval_framework.data.build_tool_policy_dataset import build_text
from agentic_eval_framework.data.validate_dataset import (
    validate_tool_policy_splits,
    validate_tool_policy_three_way,
)
from agentic_eval_framework.models.calibration import (
    expected_calibration_error,
    multiclass_brier_score,
    select_confidence_threshold,
)
from agentic_eval_framework.models.model_manifest import write_model_manifest
from agentic_eval_framework.utils.io import ensure_dir, read_jsonl, write_json


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Torch is optional for the CPU sklearn path.
        return


def _classification_metrics(
    y_true: list[str], probabilities: np.ndarray, labels: list[str]
) -> dict[str, Any]:
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    y_idx = np.asarray([label_to_index[label] for label in y_true], dtype=int)
    pred_idx = probabilities.argmax(axis=1)
    preds = [labels[int(idx)] for idx in pred_idx]
    return {
        "macro_f1": float(
            f1_score(y_true, preds, labels=labels, average="macro", zero_division=0)
        ),
        "classification_report": classification_report(
            y_true, preds, labels=labels, zero_division=0, output_dict=True
        ),
        "log_loss": float(log_loss(y_true, probabilities, labels=labels)),
        "ece": expected_calibration_error(probabilities, y_idx),
        "multiclass_brier": multiclass_brier_score(probabilities, y_idx),
    }


def train_sklearn(
    train_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    train_path: str | Path,
    calibration_path: str | Path,
    eval_path: str | Path,
    seed: int = 42,
) -> dict[str, Any]:
    import joblib

    out = ensure_dir(out_dir)
    X_train = [build_text(r) for r in train_rows]
    y_train = [r["tool_label"] for r in train_rows]
    X_calibration = [build_text(r) for r in calibration_rows]
    y_calibration = [r["tool_label"] for r in calibration_rows]
    X_eval = [build_text(r) for r in eval_rows]
    y_eval = [r["tool_label"] for r in eval_rows]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    Xtr = vectorizer.fit_transform(X_train)
    base_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    class_counts = {label: y_train.count(label) for label in set(y_train)}
    min_class_count = min(class_counts.values()) if class_counts else 0
    calibrated = min_class_count >= 3 and len(class_counts) > 1
    if calibrated:
        clf = CalibratedClassifierCV(
            estimator=base_clf, method="sigmoid", cv=min(3, min_class_count), n_jobs=1
        )
    else:
        clf = base_clf
    clf.fit(Xtr, y_train)

    labels = list(clf.classes_)
    calibration_probabilities = clf.predict_proba(vectorizer.transform(X_calibration))
    calibration_indices = np.asarray([labels.index(label) for label in y_calibration], dtype=int)
    selective = select_confidence_threshold(calibration_probabilities, calibration_indices)
    calibration_metrics = _classification_metrics(
        y_calibration, calibration_probabilities, labels
    )
    eval_probabilities = clf.predict_proba(vectorizer.transform(X_eval))
    final_metrics = _classification_metrics(y_eval, eval_probabilities, labels)

    metrics = {
        "backend": "sklearn",
        "calibrated": calibrated,
        "calibration_method": "sigmoid_cv" if calibrated else "none_small_sample",
        "threshold_selection_split": (
            "eval_compatibility_fallback"
            if Path(calibration_path).resolve() == Path(eval_path).resolve()
            else "calibration"
        ),
        "n_train": len(train_rows),
        "n_calibration": len(calibration_rows),
        "n_eval": len(eval_rows),
        "selective_policy": selective,
        "calibration_metrics": calibration_metrics,
        "labels": labels,
        **final_metrics,
    }
    joblib.dump(vectorizer, out / "vectorizer.joblib")
    joblib.dump(clf, out / "classifier.joblib")
    write_json(
        out / "model_meta.json",
        {"backend": "sklearn", "label_to_id": {label: i for i, label in enumerate(labels)}},
    )
    write_json(out / "metrics.json", metrics)
    write_model_manifest(
        out,
        backend="sklearn",
        train_path=train_path,
        calibration_path=calibration_path,
        eval_path=eval_path,
        labels=labels,
        training_config={
            "seed": seed,
            "vectorizer": "tfidf_1_2gram",
            "classifier": "logistic_regression",
            "calibrated": calibrated,
            "threshold_selection_split": (
            "eval_compatibility_fallback"
            if Path(calibration_path).resolve() == Path(eval_path).resolve()
            else "calibration"
        ),
        },
        recommended_min_confidence=selective["threshold"],
    )
    return metrics


def train_transformer(
    train_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    out_dir: str | Path,
    model_name: str,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    train_path: str | Path = "data/processed/tool_policy_train.jsonl",
    calibration_path: str | Path = "data/processed/tool_policy_calibration.jsonl",
    eval_path: str | Path = "data/processed/tool_policy_eval.jsonl",
    seed: int = 42,
) -> dict[str, Any]:  # pragma: no cover - optional heavy path
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise RuntimeError(
            "Transformer backend requires torch, transformers, datasets, and accelerate."
        ) from exc

    out = ensure_dir(out_dir)
    labels = sorted(
        set(r["tool_label"] for r in train_rows + calibration_rows + eval_rows)
    )
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def to_dataset(rows: list[dict[str, Any]]) -> Dataset:
        return Dataset.from_list(
            [{"text": build_text(r), "label": label_to_id[r["tool_label"]]} for r in rows]
        )

    def prepare(rows: list[dict[str, Any]]) -> Dataset:
        dataset = to_dataset(rows)
        dataset = dataset.map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=256),
            batched=True,
        )
        dataset = dataset.rename_column("label", "labels")
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return dataset

    train_ds = prepare(train_rows)
    calibration_ds = prepare(calibration_rows)
    eval_ds = prepare(eval_rows)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training transformer backend on device={device}")

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        return {
            "macro_f1": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            )
        }

    bf16_enabled = bool(
        torch.cuda.is_available()
        and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    fp16_enabled = bool(torch.cuda.is_available() and not bf16_enabled)
    training_kwargs = dict(
        output_dir=str(out / "hf_trainer"),
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to="none",
        save_total_limit=1,
        gradient_accumulation_steps=max(1, int(gradient_accumulation_steps)),
        gradient_checkpointing=bool(gradient_checkpointing),
        seed=seed,
        data_seed=seed,
        fp16=fp16_enabled,
        bf16=bf16_enabled,
    )
    try:
        args = TrainingArguments(**training_kwargs, eval_strategy="epoch")
    except TypeError:
        args = TrainingArguments(**training_kwargs, evaluation_strategy="epoch")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=calibration_ds,
        **({"processing_class": tokenizer} if "processing_class" in inspect.signature(Trainer.__init__).parameters else {"tokenizer": tokenizer}),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    def probabilities_for(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
        output = trainer.predict(dataset)
        logits = np.asarray(output.predictions)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        return probabilities, np.asarray(output.label_ids, dtype=int)

    calibration_probabilities, calibration_indices = probabilities_for(calibration_ds)
    selective = select_confidence_threshold(calibration_probabilities, calibration_indices)
    calibration_labels = [labels[int(idx)] for idx in calibration_indices]
    calibration_metrics = _classification_metrics(
        calibration_labels, calibration_probabilities, labels
    )
    eval_probabilities, eval_indices = probabilities_for(eval_ds)
    eval_labels = [labels[int(idx)] for idx in eval_indices]
    final_metrics = _classification_metrics(eval_labels, eval_probabilities, labels)

    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    metrics = {
        "backend": "transformer",
        "model_name": model_name,
        "device": device,
        "mixed_precision": "bf16" if bf16_enabled else ("fp16" if fp16_enabled else "fp32"),
        "gradient_accumulation_steps": max(1, int(gradient_accumulation_steps)),
        "gradient_checkpointing": bool(gradient_checkpointing),
        "threshold_selection_split": (
            "eval_compatibility_fallback"
            if Path(calibration_path).resolve() == Path(eval_path).resolve()
            else "calibration"
        ),
        "n_train": len(train_rows),
        "n_calibration": len(calibration_rows),
        "n_eval": len(eval_rows),
        "labels": labels,
        "selective_policy": selective,
        "calibration_metrics": calibration_metrics,
        **final_metrics,
    }
    write_json(out / "model_meta.json", {"backend": "transformer", "label_to_id": label_to_id})
    write_json(out / "metrics.json", metrics)
    write_model_manifest(
        out,
        backend="transformer",
        train_path=train_path,
        calibration_path=calibration_path,
        eval_path=eval_path,
        labels=labels,
        training_config={
            "seed": seed,
            "model_name": model_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": max(1, int(gradient_accumulation_steps)),
            "gradient_checkpointing": bool(gradient_checkpointing),
            "mixed_precision": metrics["mixed_precision"],
            "threshold_selection_split": (
            "eval_compatibility_fallback"
            if Path(calibration_path).resolve() == Path(eval_path).resolve()
            else "calibration"
        ),
        },
        recommended_min_confidence=selective["threshold"],
    )
    return metrics


def train_tool_policy(
    train_path: str | Path = "data/processed/tool_policy_train.jsonl",
    eval_path: str | Path = "data/processed/tool_policy_eval.jsonl",
    out_dir: str | Path = "models/tool_policy",
    backend: str = "sklearn",
    model_name: str = "distilbert-base-uncased",
    epochs: int = 2,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    *,
    calibration_path: str | Path | None = None,
) -> dict[str, Any]:
    seed = 42
    set_seed(seed)
    if calibration_path is None:
        default_calibration = Path("data/processed/tool_policy_calibration.jsonl")
        default_eval = Path("data/processed/tool_policy_eval.jsonl")
        calibration_path = (
            default_calibration
            if Path(eval_path) == default_eval and default_calibration.exists()
            else eval_path
        )
    calibration_path = Path(calibration_path)
    same_calibration_and_eval = calibration_path.resolve() == Path(eval_path).resolve()
    validation = (
        validate_tool_policy_splits(train_path, eval_path)
        if same_calibration_and_eval
        else validate_tool_policy_three_way(train_path, calibration_path, eval_path)
    )
    if validation["status"] != "PASS":
        raise ValueError(f"Dataset validation failed: {validation['reasons']}")
    train_rows = read_jsonl(train_path)
    calibration_rows = read_jsonl(calibration_path)
    eval_rows = read_jsonl(eval_path)
    if not train_rows or not calibration_rows or not eval_rows:
        raise ValueError(
            "Training/calibration/eval rows are missing. Run scripts/01_build_dataset.py first."
        )
    if backend == "transformer":
        return train_transformer(
            train_rows,
            calibration_rows,
            eval_rows,
            out_dir,
            model_name,
            epochs,
            batch_size,
            gradient_accumulation_steps,
            gradient_checkpointing,
            train_path=train_path,
            calibration_path=calibration_path,
            eval_path=eval_path,
            seed=seed,
        )
    return train_sklearn(
        train_rows,
        calibration_rows,
        eval_rows,
        out_dir,
        train_path=train_path,
        calibration_path=calibration_path,
        eval_path=eval_path,
        seed=seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/processed/tool_policy_train.jsonl")
    parser.add_argument(
        "--calibration-path", default="data/processed/tool_policy_calibration.jsonl"
    )
    parser.add_argument("--eval-path", default="data/processed/tool_policy_eval.jsonl")
    parser.add_argument("--out-dir", default="models/tool_policy")
    parser.add_argument("--backend", choices=["sklearn", "transformer"], default="sklearn")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    metrics = train_tool_policy(
        args.train_path,
        args.eval_path,
        args.out_dir,
        args.backend,
        args.model_name,
        args.epochs,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.gradient_checkpointing,
        calibration_path=args.calibration_path,
    )
    print(json.dumps(metrics, indent=2))
