from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_SPLITS = ("train", "dev", "test")


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sgd_raw_dataset(
    raw_dir: str | Path,
    *,
    required_splits: tuple[str, ...] = REQUIRED_SPLITS,
    inspect_json: bool = True,
) -> dict[str, Any]:
    """Validate the raw SGD directory contract without mutating the dataset.

    Expected layout::

        raw_dir/train/schema.json
        raw_dir/train/dialogues_*.json
        raw_dir/dev/schema.json
        raw_dir/dev/dialogues_*.json
        raw_dir/test/schema.json
        raw_dir/test/dialogues_*.json
    """
    root = Path(raw_dir).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    split_stats: dict[str, Any] = {}
    total_dialogue_files = 0
    total_dialogues = 0

    if not root.exists():
        return {
            "status": "FAIL",
            "raw_dir": str(root),
            "errors": [f"Raw dataset directory does not exist: {root}"],
            "warnings": [],
            "splits": {},
        }

    for split in required_splits:
        split_dir = root / split
        schema_path = split_dir / "schema.json"
        dialogue_files = sorted(split_dir.glob("dialogues_*.json")) if split_dir.exists() else []
        split_errors: list[str] = []
        service_count = 0
        dialogue_count = 0

        if not split_dir.is_dir():
            split_errors.append(f"Missing split directory: {split_dir}")
        if not schema_path.is_file():
            split_errors.append(f"Missing schema file: {schema_path}")
        if not dialogue_files:
            split_errors.append(f"No dialogues_*.json files found in {split_dir}")

        if inspect_json and schema_path.is_file():
            try:
                payload = json.loads(schema_path.read_text(encoding="utf-8"))
                if not isinstance(payload, list):
                    split_errors.append(f"schema.json must contain a JSON list: {schema_path}")
                else:
                    service_count = len(payload)
                    malformed = [idx for idx, row in enumerate(payload) if not isinstance(row, dict) or "service_name" not in row]
                    if malformed:
                        split_errors.append(f"Malformed schema entries in {schema_path}: first indices {malformed[:5]}")
            except Exception as exc:
                split_errors.append(f"Invalid JSON in {schema_path}: {exc}")

        if inspect_json:
            for file_path in dialogue_files:
                try:
                    payload = json.loads(file_path.read_text(encoding="utf-8"))
                    if not isinstance(payload, list):
                        split_errors.append(f"Dialogue file must contain a JSON list: {file_path}")
                        continue
                    dialogue_count += len(payload)
                    malformed = [idx for idx, row in enumerate(payload[:100]) if not isinstance(row, dict) or "turns" not in row]
                    if malformed:
                        split_errors.append(f"Malformed dialogues in {file_path}: first indices {malformed[:5]}")
                except Exception as exc:
                    split_errors.append(f"Invalid JSON in {file_path}: {exc}")

        errors.extend(split_errors)
        total_dialogue_files += len(dialogue_files)
        total_dialogues += dialogue_count
        split_stats[split] = {
            "split_dir": str(split_dir),
            "schema_present": schema_path.is_file(),
            "schema_sha256": _sha256(schema_path) if schema_path.is_file() else None,
            "service_count": service_count,
            "dialogue_file_count": len(dialogue_files),
            "dialogue_count": dialogue_count,
            "errors": split_errors,
        }

    if total_dialogues == 0 and inspect_json:
        warnings.append("No dialogues were counted; verify that files are non-empty JSON arrays.")

    return {
        "status": "PASS" if not errors else "FAIL",
        "raw_dir": str(root),
        "required_splits": list(required_splits),
        "total_dialogue_files": total_dialogue_files,
        "total_dialogues": total_dialogues,
        "errors": errors,
        "warnings": warnings,
        "splits": split_stats,
    }
