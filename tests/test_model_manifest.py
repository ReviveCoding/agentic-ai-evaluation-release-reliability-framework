from __future__ import annotations

import json

from agentic_eval_framework.data.build_tool_policy_dataset import build_tool_policy_dataset
from agentic_eval_framework.data.parse_sgd import create_sample_raw
from agentic_eval_framework.models.train_tool_policy import train_tool_policy
from agentic_eval_framework.models.transformer_tool_policy import ToolPolicyModel


def test_training_writes_reproducibility_manifest(tmp_path):
    raw = tmp_path / "raw"
    create_sample_raw(raw)
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    build_tool_policy_dataset(raw, "train", train)
    build_tool_policy_dataset(raw, "dev", evaluation)
    model_dir = tmp_path / "model"
    train_tool_policy(train, evaluation, model_dir, backend="sklearn")
    manifest = json.loads((model_dir / "model_manifest.json").read_text())
    assert len(manifest["model_fingerprint"]) == 64
    assert len(manifest["train_data"]["sha256"]) == 64
    loaded = ToolPolicyModel(model_dir)
    assert loaded.model_fingerprint == manifest["model_fingerprint"]
    assert loaded.recommended_min_confidence is not None
