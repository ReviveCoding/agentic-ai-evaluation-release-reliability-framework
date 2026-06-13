from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from agentic_eval_framework.data.build_tool_policy_dataset import service_to_tool
from agentic_eval_framework.retrieval.bm25_retriever import tokenize

CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
TERMINAL_CUES = {
    "all", "done", "goodbye", "bye", "nothing", "thanks", "thank", "cancel", "later",
}
ACTION_CUES = {
    "book", "buy", "create", "find", "get", "lookup", "play", "rent", "reserve", "search",
    "schedule", "transfer", "weather",
}
FEATURE_NAMES = [
    "fused_score",
    "recency_rrf",
    "current_rrf",
    "global_rrf",
    "service_rrf",
    "tool_compatible",
    "intent_doc",
    "service_doc",
    "token_overlap",
    "intent_overlap",
    "service_overlap",
    "terminal_service_match",
    "action_intent_match",
]


def split_identifier(value: str) -> list[str]:
    spaced = CAMEL_RE.sub(" ", str(value or "")).replace("_", " ")
    return tokenize(spaced)


def service_base(service: str) -> str:
    return str(service or "").split("_")[0].lower()




def infer_query_tool(text: str) -> str | None:
    tokens = set(tokenize(text))
    if tokens & {"weather", "forecast", "rain", "wind", "humidity", "temperature"}:
        return "weather_lookup"
    if tokens & {"movie", "media", "music", "song", "album", "watch", "listen", "play"}:
        return "media_search"
    return None

def doc_tool_label(doc: dict[str, Any]) -> str:
    return service_to_tool(
        str(doc.get("service", "")),
        str(doc.get("intent", "")),
        [],
    )


def _rrf(rank: int | None, k: int = 60) -> float:
    return 0.0 if rank is None else 1.0 / (k + int(rank))


def _overlap(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def build_feature_vector(
    *,
    doc: dict[str, Any],
    query: str,
    current_query: str,
    tool_name: str | None,
    fused_score: float,
    recency_rank: int | None,
    current_rank: int | None,
    global_rank: int | None,
    service_rank: int | None,
) -> list[float]:
    query_tokens = tokenize(query)
    current_tokens = tokenize(current_query)
    doc_tokens = tokenize(str(doc.get("title", "")) + " " + str(doc.get("text", "")))
    intent_tokens = split_identifier(str(doc.get("intent", "")))
    service_tokens = split_identifier(service_base(str(doc.get("service", ""))))
    terminal = bool(set(current_tokens) & TERMINAL_CUES)
    action = bool(set(current_tokens) & ACTION_CUES)
    doc_type = str(doc.get("doc_type", ""))
    compatible = bool(tool_name) and doc_tool_label(doc) == tool_name
    return [
        float(fused_score),
        _rrf(recency_rank),
        _rrf(current_rank),
        _rrf(global_rank),
        _rrf(service_rank),
        float(compatible),
        float(doc_type == "intent"),
        float(doc_type == "service"),
        _overlap(current_tokens or query_tokens, doc_tokens),
        _overlap(current_tokens or query_tokens, intent_tokens),
        _overlap(current_tokens or query_tokens, service_tokens),
        float(terminal and doc_type == "service"),
        float(action and doc_type == "intent"),
    ]


@dataclass
class LightweightIntentReranker:
    classifier: Any
    feature_names: list[str]

    @classmethod
    def load(cls, path: str | Path) -> "LightweightIntentReranker | None":
        artifact = Path(path)
        if not artifact.exists():
            return None
        payload = joblib.load(artifact)
        return cls(
            classifier=payload["classifier"],
            feature_names=list(payload.get("feature_names", FEATURE_NAMES)),
        )

    def score(self, vectors: list[list[float]]) -> list[float]:
        if not vectors:
            return []
        matrix = np.asarray(vectors, dtype=float)
        probabilities = self.classifier.predict_proba(matrix)
        positive_index = list(self.classifier.classes_).index(1)
        return probabilities[:, positive_index].astype(float).tolist()
