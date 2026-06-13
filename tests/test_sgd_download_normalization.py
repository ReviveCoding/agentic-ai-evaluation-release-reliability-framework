import json
from pathlib import Path

from agentic_eval_framework.data.download_sgd import normalize_extracted_sgd


def test_normalize_extracted_sgd_nested_layout(tmp_path):
    nested = tmp_path / "extract" / "dstc8-schema-guided-dialogue-master"
    train = nested / "train"
    dev = nested / "dev"
    train.mkdir(parents=True)
    dev.mkdir(parents=True)
    for split in [train, dev]:
        (split / "schema.json").write_text(json.dumps([]), encoding="utf-8")
        (split / "dialogues_001.json").write_text(json.dumps([]), encoding="utf-8")
    out = normalize_extracted_sgd(tmp_path / "extract", tmp_path / "raw" / "sgd")
    assert (out / "train" / "schema.json").exists()
    assert (out / "dev" / "dialogues_001.json").exists()
