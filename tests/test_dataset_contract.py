from pathlib import Path

from agentic_eval_framework.data.dataset_contract import validate_sgd_raw_dataset
from agentic_eval_framework.data.parse_sgd import create_sample_raw


def test_external_dataset_contract_passes(tmp_path: Path) -> None:
    raw = tmp_path / "external_sgd"
    create_sample_raw(raw)
    result = validate_sgd_raw_dataset(raw)
    assert result["status"] == "PASS"
    assert result["total_dialogue_files"] == 3
    assert result["total_dialogues"] == 18
    assert set(result["splits"]) == {"train", "dev", "test"}


def test_external_dataset_contract_fails_missing_split(tmp_path: Path) -> None:
    raw = tmp_path / "external_sgd"
    create_sample_raw(raw)
    for child in (raw / "test").iterdir():
        child.unlink()
    (raw / "test").rmdir()
    result = validate_sgd_raw_dataset(raw)
    assert result["status"] == "FAIL"
    assert any("Missing split directory" in error for error in result["errors"])
