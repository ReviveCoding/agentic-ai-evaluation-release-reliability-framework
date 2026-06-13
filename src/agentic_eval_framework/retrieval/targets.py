from __future__ import annotations

from typing import Any, Iterable

from agentic_eval_framework.retrieval.recency_query import recency_weighted_query


def service_base(service: str) -> str:
    return str(service or "").split("_")[0].lower()


def service_doc_id(service: str) -> str:
    return f"service::{service}"


def intent_doc_id(service: str, intent: str) -> str:
    return f"intent::{service}::{intent}"


def target_doc_id(payload: dict[str, Any], available_doc_ids: set[str] | None = None) -> str:
    service = str(payload.get("service", ""))
    intent = str(payload.get("intent", ""))
    candidate = intent_doc_id(service, intent) if service and intent else service_doc_id(service)
    if available_doc_ids is None or candidate in available_doc_ids:
        return candidate
    return service_doc_id(service)


def compatible_doc_ids(payload: dict[str, Any], docs: Iterable[dict[str, Any]]) -> set[str]:
    """Return semantically compatible evidence IDs without affecting ranking.

    SGD service versions such as Hotels_1 and Hotels_4 can expose the same
    semantic action. Exact service-version recovery is reported separately, but
    grounding accepts an intent document from another version with the same
    service base and intent name, or a service document with the same base.
    """

    expected_service = str(payload.get("service", ""))
    expected_base = service_base(expected_service)
    expected_intent = str(payload.get("intent", ""))
    compatible: set[str] = set()
    for doc in docs:
        doc_service = str(doc.get("service", ""))
        if service_base(doc_service) != expected_base:
            continue
        doc_type = str(doc.get("doc_type", ""))
        doc_intent = str(doc.get("intent", ""))
        if doc_type == "service" or (doc_type == "intent" and doc_intent == expected_intent):
            compatible.add(str(doc.get("doc_id", "")))
    return {doc_id for doc_id in compatible if doc_id}


def build_retrieval_query(payload: dict[str, Any], mode: str = "recency_weighted") -> str:
    if mode == "recency_weighted":
        return recency_weighted_query(payload)
    parts = [
        str(payload.get("dialogue_context", "")),
        str(payload.get("user_request", payload.get("user_utterance", ""))),
    ]
    known_slots = payload.get("known_slots", {}) or {}
    if isinstance(known_slots, dict):
        parts.extend(str(value) for value in known_slots.values() if value not in (None, ""))
    if mode == "metadata_enriched":
        parts.extend(
            [
                str(payload.get("observed_service", payload.get("service", ""))),
                str(payload.get("observed_intent", payload.get("intent", ""))),
            ]
        )
    elif mode != "utterance_context":
        raise ValueError(
            "retrieval query mode must be recency_weighted, utterance_context, or metadata_enriched"
        )
    return " ".join(part for part in parts if part).strip()
