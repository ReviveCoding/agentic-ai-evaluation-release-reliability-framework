from agentic_eval_framework.data.build_golden_trajectories import build_golden_trajectories
from agentic_eval_framework.data.build_tool_policy_dataset import build_tool_policy_dataset
from agentic_eval_framework.data.parse_sgd import create_sample_raw


def test_dataset_builder(tmp_path):
    raw = tmp_path / "raw" / "sgd"
    create_sample_raw(raw)
    rows = build_tool_policy_dataset(raw, "train", tmp_path / "train.jsonl")
    assert rows
    assert {r["tool_label"] for r in rows} >= {"search_places", "calendar_lookup", "ask_clarification", "safety_check"}
    traj = build_golden_trajectories(tmp_path / "train.jsonl", tmp_path / "golden.jsonl")
    assert len(traj) == len(rows)
    assert all("expected_tools" in t for t in traj)
