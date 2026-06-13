from __future__ import annotations

from agentic_eval_framework.data.validate_dataset import validate_tool_policy_splits
from agentic_eval_framework.utils.io import write_jsonl


def _row(example_id: str, label: str) -> dict:
    return {
        "example_id": example_id,
        "user_utterance": "request",
        "service": "Service_1",
        "intent": "Intent",
        "tool_label": label,
    }


def test_dataset_validation_detects_unseen_eval_label(tmp_path):
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    write_jsonl(train, [_row("train_1", "search_docs")])
    write_jsonl(evaluation, [_row("eval_1", "weather_lookup")])
    result = validate_tool_policy_splits(train, evaluation)
    assert result["status"] == "FAIL"
    assert result["unseen_eval_labels"] == ["weather_lookup"]
