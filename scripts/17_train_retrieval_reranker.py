from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agentic_eval_framework.retrieval.hierarchical_retriever import HierarchicalBM25Retriever
from agentic_eval_framework.retrieval.recency_query import current_utterance
from agentic_eval_framework.retrieval.targets import build_retrieval_query, compatible_doc_ids, target_doc_id
from agentic_eval_framework.retrieval.top1_reranker import FEATURE_NAMES
from agentic_eval_framework.utils.io import read_jsonl, write_json


def row_to_payload(row: dict) -> dict:
    return {
        "scenario_id": row.get("example_id", row.get("source_dialogue_id", "train")),
        "dialogue_context": row.get("dialogue_context", ""),
        "user_request": row.get("user_utterance", ""),
        "known_slots": row.get("known_slots", {}),
        "service": row.get("service", ""),
        "intent": row.get("intent", ""),
        "expected_tools": [row.get("tool_label", "")],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight pointwise retrieval reranker.")
    parser.add_argument("--train-path", default="data/processed/tool_policy_train.jsonl")
    parser.add_argument("--docs-path", default="data/processed/service_docs.jsonl")
    parser.add_argument("--out-path", default="models/retrieval_reranker.joblib")
    parser.add_argument("--metrics-path", default="outputs/retrieval_reranker_training.json")
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(args.train_path)
    docs = [doc for doc in read_jsonl(args.docs_path) if doc.get("doc_type") in {"service", "intent"}]
    if not rows or not docs:
        raise SystemExit("Training rows and service/intent documents are required.")
    rows = rows[: args.max_rows] if args.max_rows > 0 else rows
    available_ids = {str(doc.get("doc_id")) for doc in docs}
    retriever = HierarchicalBM25Retriever(docs, reranker_path="__missing_reranker__.joblib")

    features: list[list[float]] = []
    labels: list[int] = []
    groups = 0
    positive_in_candidates = 0
    for row in rows:
        payload = row_to_payload(row)
        tool_name = str(row.get("tool_label", ""))
        if tool_name not in {"search_places", "calendar_lookup", "media_search", "weather_lookup", "search_docs"}:
            continue
        query = build_retrieval_query(payload, mode="recency_weighted")
        results = retriever.search(
            query,
            top_k=args.candidate_k,
            tool_name=tool_name,
            current_query=current_utterance(payload),
            use_reranker=False,
            include_features=True,
        )
        compatible = compatible_doc_ids(payload, docs)
        target = target_doc_id(payload, available_ids)
        compatible.add(target)
        has_positive = any(str(item.get("doc_id")) in compatible for item in results)
        positive_in_candidates += int(has_positive)
        groups += 1
        for item in results:
            vector = list(item.get("reranker_features") or [])
            if len(vector) != len(FEATURE_NAMES):
                raise RuntimeError("Candidate is missing reranker features")
            features.append(vector)
            labels.append(int(str(item.get("doc_id")) in compatible))

    if not features or len(set(labels)) < 2:
        raise SystemExit("Insufficient positive/negative reranker examples.")
    matrix = np.asarray(features, dtype=float)
    target = np.asarray(labels, dtype=int)
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=args.seed,
        solver="liblinear",
    )
    classifier.fit(matrix, target)
    probability = classifier.predict_proba(matrix)[:, list(classifier.classes_).index(1)]
    prediction = (probability >= 0.5).astype(int)
    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "classifier": classifier,
            "feature_names": FEATURE_NAMES,
            "training_rows": groups,
            "candidate_k": args.candidate_k,
            "version": "top1-reranker-v1",
        },
        out,
    )
    metrics = {
        "training_rows": groups,
        "candidate_examples": len(labels),
        "class_counts": dict(Counter(labels)),
        "positive_candidate_coverage": positive_in_candidates / max(1, groups),
        "roc_auc_train": roc_auc_score(target, probability),
        "classification_report_train": classification_report(target, prediction, output_dict=True, zero_division=0),
        "feature_names": FEATURE_NAMES,
        "coefficients": {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, classifier.coef_[0], strict=True)
        },
        "artifact": str(out),
    }
    write_json(args.metrics_path, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
