import json

from agentic_eval_framework.data.build_tool_policy_dataset import build_tool_policy_dataset
from agentic_eval_framework.data.parse_sgd import create_sample_raw
from agentic_eval_framework.data.validate_dataset import validate_tool_policy_three_way
from agentic_eval_framework.models.train_tool_policy import train_tool_policy


def test_calibration_and_final_eval_are_disjoint_and_manifested(tmp_path):
    raw = tmp_path / "raw"
    create_sample_raw(raw)
    train = tmp_path / "train.jsonl"
    calibration = tmp_path / "calibration.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    build_tool_policy_dataset(raw, "train", train)
    build_tool_policy_dataset(raw, "dev", calibration)
    build_tool_policy_dataset(raw, "test", evaluation)
    validation = validate_tool_policy_three_way(train, calibration, evaluation)
    assert validation["status"] == "PASS"
    assert validation["calibration_eval_id_overlap"] == []

    model_dir = tmp_path / "model"
    metrics = train_tool_policy(
        train, evaluation, model_dir, backend="sklearn", calibration_path=calibration
    )
    manifest = json.loads((model_dir / "model_manifest.json").read_text())
    assert metrics["threshold_selection_split"] == "calibration"
    assert manifest["calibration_data"]["sha256"] != manifest["eval_data"]["sha256"]
    assert manifest["training_config"]["threshold_selection_split"] == "calibration"
