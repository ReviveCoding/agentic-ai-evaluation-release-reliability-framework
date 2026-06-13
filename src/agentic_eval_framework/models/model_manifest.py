from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentic_eval_framework.utils.fingerprints import package_versions, sha256_file, sha256_paths, stable_json_hash
from agentic_eval_framework.utils.io import write_json

ARTIFACT_PATTERNS = (
    "model_meta.json",
    "metrics.json",
    "vectorizer.joblib",
    "classifier.joblib",
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
)


def _artifact_paths(model_dir: Path) -> list[Path]:
    return [model_dir / name for name in ARTIFACT_PATTERNS if (model_dir / name).is_file()]


def write_model_manifest(
    model_dir: str | Path,
    *,
    backend: str,
    train_path: str | Path,
    eval_path: str | Path,
    calibration_path: str | Path | None = None,
    labels: list[str],
    training_config: dict[str, Any],
    recommended_min_confidence: float | None = None,
) -> dict[str, Any]:
    out = Path(model_dir)
    manifest = {
        "schema_version": 2,
        "backend": backend,
        "labels": labels,
        "training_config": training_config,
        "train_data": {"path": str(train_path), "sha256": sha256_file(train_path)},
        "calibration_data": (
            {"path": str(calibration_path), "sha256": sha256_file(calibration_path)}
            if calibration_path is not None else None
        ),
        "eval_data": {"path": str(eval_path), "sha256": sha256_file(eval_path)},
        "environment": package_versions(
            ["numpy", "scikit-learn", "torch", "transformers", "datasets", "accelerate"]
        ),
        "recommended_min_confidence": recommended_min_confidence,
    }
    manifest["artifact_sha256"] = sha256_paths(_artifact_paths(out))
    # The identity hash intentionally excludes filesystem paths so the same
    # trained artifact keeps the same fingerprint after cloning or packaging.
    fingerprint_payload = {
        "schema_version": manifest["schema_version"],
        "backend": backend,
        "labels": labels,
        "training_config": training_config,
        "train_data_sha256": manifest["train_data"]["sha256"],
        "calibration_data_sha256": (manifest["calibration_data"] or {}).get("sha256"),
        "eval_data_sha256": manifest["eval_data"]["sha256"],
        "artifact_sha256": manifest["artifact_sha256"],
        "recommended_min_confidence": recommended_min_confidence,
    }
    manifest["model_fingerprint"] = stable_json_hash(fingerprint_payload)
    write_json(out / "model_manifest.json", manifest)
    return manifest


def load_model_manifest(model_dir: str | Path) -> dict[str, Any]:
    path = Path(model_dir) / "model_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
