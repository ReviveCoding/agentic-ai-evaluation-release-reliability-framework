from __future__ import annotations

from collections import Counter, defaultdict
import random
from typing import Any, Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from agentic_eval_framework.data.build_tool_policy_dataset import build_text


ABLATION_MODES = (
    "full",
    "utterance_context_only",
    "schema_descriptions_only",
    "metadata_names_only",
    "slots_risk_only",
    "no_oracle_structured",
)


def build_ablation_text(row: dict[str, Any], mode: str) -> str:
    if mode == "full":
        return build_text(row)
    if mode == "utterance_context_only":
        return (
            f"[CONTEXT] {row.get('dialogue_context', '')} "
            f"[USER] {row.get('user_utterance', '')}"
        )
    if mode == "schema_descriptions_only":
        return (
            f"[SERVICE_DESCRIPTION] {row.get('service_description', '')} "
            f"[INTENT_DESCRIPTION] {row.get('intent_description', '')}"
        )
    if mode == "metadata_names_only":
        return f"[SERVICE] {row.get('service', '')} [INTENT] {row.get('intent', '')}"
    if mode == "slots_risk_only":
        return (
            f"[SLOTS] required={row.get('required_slots', [])} "
            f"known={row.get('known_slots', {})} missing={row.get('missing_slots', [])} "
            f"[RISK] {row.get('risk_flags', [])}"
        )
    if mode == "no_oracle_structured":
        return (
            f"[CONTEXT] {row.get('dialogue_context', '')} "
            f"[USER] {row.get('user_utterance', '')} "
            f"[SERVICE_DESCRIPTION] {row.get('service_description', '')} "
            f"[INTENT_DESCRIPTION] {row.get('intent_description', '')}"
        )
    raise ValueError(f"Unknown ablation mode: {mode}")


def _metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0, output_dict=True
        ),
    }


def evaluate_transformer_input_modes(
    rows: list[dict[str, Any]],
    model_dir: str,
    modes: tuple[str, ...] = ABLATION_MODES,
    batch_size: int = 64,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    id_to_label = {int(key): value for key, value in model.config.id2label.items()}
    labels = sorted(set(str(row["tool_label"]) for row in rows))
    y_true = [str(row["tool_label"]) for row in rows]

    output: dict[str, Any] = {}
    for mode in modes:
        texts = [build_ablation_text(row, mode) for row in rows]
        predictions: list[str] = []
        confidences: list[float] = []
        for start in range(0, len(texts), max(1, int(batch_size))):
            chunk = texts[start : start + max(1, int(batch_size))]
            batch = tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.inference_mode():
                probabilities = torch.softmax(model(**batch).logits, dim=-1)
            confidence, predicted = probabilities.max(dim=-1)
            predictions.extend(id_to_label[int(index)] for index in predicted.detach().cpu().tolist())
            confidences.extend(float(value) for value in confidence.detach().cpu().tolist())
        result = _metrics(y_true, predictions, labels)
        result["mean_confidence"] = float(np.mean(confidences)) if confidences else 0.0
        output[mode] = result

    full_score = float(output.get("full", {}).get("macro_f1", 0.0))
    for mode, result in output.items():
        result["macro_f1_drop_vs_full"] = full_score - float(result["macro_f1"])
    return {"device": str(device), "modes": output}


def evaluate_linear_retrain_modes(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    modes: tuple[str, ...] = ABLATION_MODES,
    seed: int = 42,
) -> dict[str, Any]:
    labels = sorted(set(str(row["tool_label"]) for row in train_rows + eval_rows))
    y_train = [str(row["tool_label"]) for row in train_rows]
    y_eval = [str(row["tool_label"]) for row in eval_rows]
    output: dict[str, Any] = {}

    for mode in modes:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        train_text = [build_ablation_text(row, mode) for row in train_rows]
        eval_text = [build_ablation_text(row, mode) for row in eval_rows]
        x_train = vectorizer.fit_transform(train_text)
        classifier = LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=seed,
        )
        classifier.fit(x_train, y_train)
        predictions = [str(value) for value in classifier.predict(vectorizer.transform(eval_text))]
        output[mode] = _metrics(y_eval, predictions, labels)

    full_score = float(output.get("full", {}).get("macro_f1", 0.0))
    for mode, result in output.items():
        result["macro_f1_drop_vs_full"] = full_score - float(result["macro_f1"])
    return output


def evaluate_metadata_oracle(
    train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    labels = sorted(set(str(row["tool_label"]) for row in train_rows + eval_rows))
    buckets: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        service_base = str(row.get("service", "")).split("_")[0].lower()
        return (
            service_base,
            str(row.get("intent", "")),
            bool(row.get("missing_slots")),
            bool(row.get("risk_flags")),
        )

    for row in train_rows:
        label = str(row["tool_label"])
        buckets[key(row)][label] += 1
        global_counts[label] += 1
    fallback = global_counts.most_common(1)[0][0]

    predictions: list[str] = []
    seen = 0
    for row in eval_rows:
        counts = buckets.get(key(row))
        if counts:
            predictions.append(counts.most_common(1)[0][0])
            seen += 1
        else:
            predictions.append(fallback)
    y_true = [str(row["tool_label"]) for row in eval_rows]
    result = _metrics(y_true, predictions, labels)
    result["key_coverage"] = seen / max(1, len(eval_rows))
    result["unique_train_keys"] = len(buckets)
    return result


def evaluate_permutation_sensitivity(
    rows: list[dict[str, Any]],
    predict_texts: Callable[[list[str]], list[str]],
    seed: int = 42,
) -> dict[str, Any]:
    """Measure score loss when structured feature groups are permuted."""

    labels = sorted(set(str(row["tool_label"]) for row in rows))
    y_true = [str(row["tool_label"]) for row in rows]
    rng = random.Random(seed)

    groups = {
        "service_intent": ("service", "intent"),
        "schema_descriptions": ("service_description", "intent_description"),
        "slots_risk": ("required_slots", "known_slots", "missing_slots", "risk_flags"),
        "utterance_context": ("user_utterance", "dialogue_context"),
    }
    baseline_predictions = predict_texts([build_text(row) for row in rows])
    baseline = _metrics(y_true, baseline_predictions, labels)
    output: dict[str, Any] = {"baseline": baseline, "groups": {}}

    for name, fields in groups.items():
        permutation = list(range(len(rows)))
        rng.shuffle(permutation)
        modified: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            copy = dict(row)
            source = rows[permutation[index]]
            for field in fields:
                copy[field] = source.get(field)
            modified.append(copy)
        predictions = predict_texts([build_text(row) for row in modified])
        result = _metrics(y_true, predictions, labels)
        result["macro_f1_drop"] = baseline["macro_f1"] - result["macro_f1"]
        output["groups"][name] = result
    return output
