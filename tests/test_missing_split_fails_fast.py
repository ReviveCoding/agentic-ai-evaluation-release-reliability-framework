import pytest

from agentic_eval_framework.data.build_tool_policy_dataset import build_tool_policy_dataset


def test_missing_raw_split_fails_instead_of_silently_using_sample_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="split directory does not exist"):
        build_tool_policy_dataset(
            tmp_path / "missing_raw", "test", tmp_path / "out.jsonl"
        )
