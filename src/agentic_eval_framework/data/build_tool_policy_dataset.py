from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_eval_framework.data.parse_sgd import intent_required_slots, load_dialogues, load_schemas, user_frames
from agentic_eval_framework.utils.io import write_jsonl


SENSITIVE_SERVICES = {"Bank", "Payment", "Banks", "MoneyTransfer"}


def service_to_tool(service: str, intent: str, missing_slots: list[str]) -> str:
    base = service.split("_")[0].lower()
    intent_l = (intent or "").lower()
    if missing_slots:
        return "ask_clarification"
    if any(s.lower() in base for s in SENSITIVE_SERVICES) or "transfer" in intent_l or "payment" in intent_l:
        return "safety_check"
    if base in {"hotels", "hotel", "travel", "events", "restaurants", "flights", "rentalcars", "ridesharing", "buses", "trains", "homes", "attractions"}:
        return "search_places"
    if base in {"calendar", "appointment"}:
        if any(token in intent_l for token in {"create", "add", "schedule", "book", "update", "delete", "cancel"}):
            return "calendar_write"
        return "calendar_lookup"
    if base in {"weather"}:
        return "weather_lookup"
    if base in {"media", "movies", "music"}:
        return "media_search"
    return "search_docs"


def normalize_slot_values(slot_values: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in slot_values.items():
        if isinstance(v, list):
            out[k] = v[0] if v else None
        else:
            out[k] = v
    return out


def build_tool_policy_dataset(
    raw_dir: str | Path = "data/raw/sgd",
    split: str = "train",
    out_path: str | Path = "data/processed/tool_policy_train.jsonl",
    max_dialogues: int | None = None,
) -> list[dict[str, Any]]:
    split_dir = Path(raw_dir) / split
    schemas = load_schemas(split_dir)
    dialogues = load_dialogues(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(
            f"SGD split directory does not exist: {split_dir}. "
            "Download the public dataset or run with --use-sample."
        )
    if not schemas:
        raise FileNotFoundError(f"No schema.json found for SGD split: {split_dir}")
    if not dialogues:
        raise FileNotFoundError(f"No dialogues_*.json files found for SGD split: {split_dir}")
    if max_dialogues:
        dialogues = dialogues[:max_dialogues]

    rows: list[dict[str, Any]] = []
    for d in dialogues:
        context: list[str] = []
        for turn_idx, turn in enumerate(d.get("turns", [])):
            if turn.get("speaker") == "USER":
                frames = user_frames(turn)
                if not frames:
                    continue
                for frame_idx, frame in enumerate(frames):
                    service = frame.get("service", d.get("services", ["Unknown_1"])[0])
                    state = frame.get("state", {}) or {}
                    intent = state.get("active_intent", "None")
                    known_slots = normalize_slot_values(state.get("slot_values", {}) or {})
                    schema = schemas.get(service, {})
                    required = intent_required_slots(schema, intent)
                    missing = [slot for slot in required if slot not in known_slots or known_slots.get(slot) in (None, "")]
                    tool_label = service_to_tool(service, intent, missing)
                    risk_flags = ["sensitive_service"] if tool_label == "safety_check" else []
                    row = {
                        "example_id": f"{split}::{d.get('dialogue_id','dialogue')}_{turn_idx}_{frame_idx}",
                        "source_dialogue_id": d.get("dialogue_id"),
                        "dialogue_context": " ".join(context[-4:]),
                        "user_utterance": turn.get("utterance", ""),
                        "service": service,
                        "intent": intent,
                        "service_description": schema.get("description", ""),
                        "known_slots": known_slots,
                        "missing_slots": missing,
                        "required_slots": required,
                        "tool_label": tool_label,
                        "expected_next_action": "tool_call" if tool_label not in {"ask_clarification", "safety_check"} else tool_label,
                        "risk_flags": risk_flags,
                        "split": split,
                    }
                    rows.append(row)
            speaker = turn.get("speaker", "UNK")
            context.append(f"[{speaker}] {turn.get('utterance','')}")
    write_jsonl(out_path, rows)
    return rows


def build_text(row: dict[str, Any]) -> str:
    return (
        f"[CONTEXT] {row.get('dialogue_context','')} "
        f"[USER] {row.get('user_utterance','')} "
        f"[SERVICE] {row.get('service','')} {row.get('service_description','')} "
        f"[INTENT] {row.get('intent','')} {row.get('intent_description','')} "
        f"[SLOTS] required={row.get('required_slots',[])} known={row.get('known_slots',{})} missing={row.get('missing_slots',[])} "
        f"[RISK] {row.get('risk_flags',[])}"
    )
