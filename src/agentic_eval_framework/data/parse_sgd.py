from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from agentic_eval_framework.data.sample_sgd import SAMPLE_DIALOGUES, SAMPLE_SCHEMAS
from agentic_eval_framework.utils.io import ensure_dir, write_json


def create_sample_raw(raw_dir: str | Path = "data/raw/sgd") -> None:
    raw = Path(raw_dir)
    ensure_dir(raw / "train")
    ensure_dir(raw / "dev")
    ensure_dir(raw / "test")
    dev_dialogues = copy.deepcopy(SAMPLE_DIALOGUES)
    test_dialogues = copy.deepcopy(SAMPLE_DIALOGUES)
    paraphrases = {
        "sample_001": "Please look for Cambridge hotels that include free wifi.",
        "sample_002": "Show my meetings for tomorrow.",
        "sample_003": "Put lunch with Dana on my calendar.",
        "sample_004": "Check whether New York gets rain this weekend.",
        "sample_005": "Search for a sci fi movie recommendation.",
        "sample_006": "Move 500 dollars into my savings account.",
    }
    test_paraphrases = {
        "sample_001": "I need lodging near Cambridge and wifi must be included.",
        "sample_002": "Which appointments are on tomorrow's agenda?",
        "sample_003": "Add a calendar item for lunch with Dana.",
        "sample_004": "Give me the weekend rain outlook for New York.",
        "sample_005": "Recommend a movie in the science-fiction genre.",
        "sample_006": "Please send five hundred dollars to savings.",
    }
    for dialogues, mapping, suffix in (
        (dev_dialogues, paraphrases, "dev"),
        (test_dialogues, test_paraphrases, "test"),
    ):
        for dialogue in dialogues:
            dialogue["dialogue_id"] = f"{dialogue.get('dialogue_id')}_{suffix}"
            base_id = dialogue["dialogue_id"].removesuffix(f"_{suffix}")
            user_turn = next((t for t in dialogue.get("turns", []) if t.get("speaker") == "USER"), None)
            if user_turn and base_id in mapping:
                user_turn["utterance"] = mapping[base_id]
    write_json(raw / "train" / "schema.json", SAMPLE_SCHEMAS)
    write_json(raw / "train" / "dialogues_001.json", SAMPLE_DIALOGUES)
    write_json(raw / "dev" / "schema.json", SAMPLE_SCHEMAS)
    write_json(raw / "dev" / "dialogues_001.json", dev_dialogues)
    write_json(raw / "test" / "schema.json", SAMPLE_SCHEMAS)
    write_json(raw / "test" / "dialogues_001.json", test_dialogues)


def load_schemas(split_dir: str | Path) -> dict[str, dict[str, Any]]:
    schema_path = Path(split_dir) / "schema.json"
    if not schema_path.exists():
        return {}
    with schema_path.open("r", encoding="utf-8") as f:
        schemas = json.load(f)
    return {s["service_name"]: s for s in schemas}


def load_dialogues(split_dir: str | Path) -> list[dict[str, Any]]:
    split = Path(split_dir)
    files = sorted(split.glob("dialogues_*.json"))
    if not files:
        return []
    all_dialogues: list[dict[str, Any]] = []
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            all_dialogues.extend(json.load(f))
    return all_dialogues


def intent_required_slots(schema: dict[str, Any], intent_name: str) -> list[str]:
    for intent in schema.get("intents", []):
        if intent.get("name") == intent_name:
            return list(intent.get("required_slots", []))
    return []


def user_frames(turn: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all frames for a user turn.

    SGD turns can contain multiple frames when a user utterance touches multiple
    services. The first-frame helper is kept for backward-compatible tests, but
    dataset construction uses all frames to avoid silently dropping labels.
    """
    if turn.get("speaker") != "USER":
        return []
    return list(turn.get("frames") or [])


def first_user_frame(turn: dict[str, Any]) -> dict[str, Any] | None:
    frames = user_frames(turn)
    return frames[0] if frames else None
