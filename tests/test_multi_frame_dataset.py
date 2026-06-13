import json
from pathlib import Path

from agentic_eval_framework.data.build_tool_policy_dataset import build_tool_policy_dataset


def test_dataset_builder_keeps_multiple_user_frames(tmp_path):
    raw = tmp_path / "raw" / "sgd" / "train"
    raw.mkdir(parents=True)
    schemas = [
        {"service_name": "Hotels_1", "description": "hotel", "intents": [{"name": "SearchHotel", "required_slots": ["location"]}], "slots": []},
        {"service_name": "Weather_1", "description": "weather", "intents": [{"name": "GetWeather", "required_slots": ["date"]}], "slots": []},
    ]
    dialogues = [{
        "dialogue_id": "d1",
        "services": ["Hotels_1", "Weather_1"],
        "turns": [{
            "speaker": "USER",
            "utterance": "Find a hotel and check weather tomorrow",
            "frames": [
                {"service": "Hotels_1", "state": {"active_intent": "SearchHotel", "slot_values": {"location": ["Boston"]}}},
                {"service": "Weather_1", "state": {"active_intent": "GetWeather", "slot_values": {"date": ["tomorrow"]}}},
            ],
        }],
    }]
    (raw / "schema.json").write_text(json.dumps(schemas), encoding="utf-8")
    (raw / "dialogues_001.json").write_text(json.dumps(dialogues), encoding="utf-8")
    rows = build_tool_policy_dataset(tmp_path / "raw" / "sgd", split="train", out_path=tmp_path / "train.jsonl")
    assert len(rows) == 2
    assert {r["tool_label"] for r in rows} == {"search_places", "weather_lookup"}
