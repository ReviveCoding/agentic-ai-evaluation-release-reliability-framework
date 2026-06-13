from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from agentic_eval_framework.data.parse_sgd import load_dialogues, load_schemas, user_frames
from agentic_eval_framework.retrieval.targets import intent_doc_id, service_doc_id
from agentic_eval_framework.utils.io import write_jsonl


def _normalize_splits(split: str | Iterable[str]) -> list[str]:
    if isinstance(split, str):
        return [split]
    return list(split)


def build_service_docs(
    raw_dir: str | Path = "data/raw/sgd",
    split: str | Iterable[str] = ("train", "dev"),
    out_path: str | Path = "data/processed/service_docs.jsonl",
) -> list[dict[str, Any]]:
    """Build service and intent documents available to the runtime retriever.

    Intent-level documents make retrieval meaningful for services that expose
    multiple actions, such as reading versus writing calendar events. Dialogue
    labels remain held out; only public schema descriptions are indexed.
    """
    split_names = _normalize_splits(split)
    merged: dict[str, dict[str, Any]] = {}
    for split_name in split_names:
        split_dir = Path(raw_dir) / split_name
        if not split_dir.exists():
            continue
        merged.update(load_schemas(split_dir))

    # Mine a small set of training utterances as public, label-preserving retrieval
    # examples. Evaluation utterances are never indexed. This makes utterance-only
    # retrieval realistic without leaking the held-out query's service/intent.
    examples: dict[tuple[str, str], list[str]] = {}
    example_split = "train" if "train" in split_names else (split_names[0] if split_names else "train")
    example_dir = Path(raw_dir) / example_split
    if example_dir.exists():
        for dialogue in load_dialogues(example_dir):
            for turn in dialogue.get("turns", []):
                if turn.get("speaker") != "USER":
                    continue
                utterance = str(turn.get("utterance", "")).strip()
                for frame in user_frames(turn):
                    service = str(frame.get("service", ""))
                    intent = str((frame.get("state", {}) or {}).get("active_intent", ""))
                    key = (service, intent)
                    bucket = examples.setdefault(key, [])
                    if utterance and utterance not in bucket and len(bucket) < 20:
                        bucket.append(utterance)

    docs: list[dict[str, Any]] = []
    for service_name, schema in sorted(merged.items()):
        slots = {str(slot.get("name")): str(slot.get("description", "")) for slot in schema.get("slots", [])}
        slot_text = "; ".join(f"{name}: {description}" for name, description in slots.items())
        intent_text = "; ".join(
            f"{intent.get('name')}: {intent.get('description','')}" for intent in schema.get("intents", [])
        )
        docs.append({
            "doc_id": service_doc_id(service_name),
            "doc_type": "service",
            "service": service_name,
            "intent": "",
            "title": service_name,
            "text": f"{schema.get('description','')} Intents: {intent_text}. Slots: {slot_text}.",
        })
        for intent in schema.get("intents", []):
            intent_name = str(intent.get("name", ""))
            required = list(intent.get("required_slots", []) or [])
            optional = list(intent.get("optional_slots", []) or [])
            relevant = required + optional
            relevant_slots = "; ".join(f"{slot}: {slots.get(slot, '')}" for slot in relevant)
            example_text = " | ".join(examples.get((service_name, intent_name), [])[:8])
            docs.append({
                "doc_id": intent_doc_id(service_name, intent_name),
                "doc_type": "intent",
                "service": service_name,
                "intent": intent_name,
                "title": f"{service_name} {intent_name}",
                "text": (
                    f"Service: {schema.get('description','')}. Action: {intent_name}. "
                    f"{intent.get('description','')}. Required inputs: {', '.join(required) or 'none'}. "
                    f"Optional inputs: {', '.join(optional) or 'none'}. Slot definitions: {relevant_slots}. "
                    f"Example requests: {example_text}."
                ),
            })

    tool_docs = [
        ("tool::search_places", "Search hotels, restaurants, travel, events, and place-like services."),
        ("tool::calendar_lookup", "Read calendar events and calendar availability without changing state."),
        ("tool::calendar_write", "Stage a calendar create, update, or delete action for confirmation."),
        ("tool::media_search", "Search movies, music, shows, and media-like services."),
        ("tool::weather_lookup", "Retrieve weather forecasts by location and date."),
        ("tool::search_docs", "Retrieve personal or enterprise documents and policies."),
        ("tool::ask_clarification", "Ask for missing required information before taking action."),
        ("tool::safety_check", "Review sensitive or policy-constrained requests before acting."),
        ("tool::final_answer", "Complete the task with a grounded answer."),
    ]
    for doc_id, text in tool_docs:
        docs.append({"doc_id": doc_id, "doc_type": "tool", "service": "framework", "intent": "", "title": doc_id, "text": text})
    write_jsonl(out_path, docs)
    return docs
