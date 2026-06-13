from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from agentic_eval_framework.utils.io import read_jsonl, write_json


def validate_tool_policy_splits(
    train_path: str | Path,
    eval_path: str | Path,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    train = read_jsonl(train_path)
    evaluation = read_jsonl(eval_path)
    train_ids = [str(row.get("example_id", "")) for row in train]
    eval_ids = [str(row.get("example_id", "")) for row in evaluation]
    train_labels = Counter(str(row.get("tool_label", "")) for row in train)
    eval_labels = Counter(str(row.get("tool_label", "")) for row in evaluation)
    duplicate_train = sorted(k for k, count in Counter(train_ids).items() if k and count > 1)
    duplicate_eval = sorted(k for k, count in Counter(eval_ids).items() if k and count > 1)
    overlap = sorted(set(train_ids) & set(eval_ids) - {""})
    unseen_eval_labels = sorted(set(eval_labels) - set(train_labels))
    missing_required_fields: list[str] = []
    empty_observed_metadata: list[str] = []
    required_nonempty = {"example_id", "user_utterance", "tool_label"}
    required_present = {"service", "intent"}
    for split, rows in (("train", train), ("eval", evaluation)):
        for idx, row in enumerate(rows):
            absent = sorted(
                [key for key in required_nonempty if key not in row or row.get(key) in {None, ""}]
                + [key for key in required_present if key not in row]
            )
            if absent:
                missing_required_fields.append(f"{split}[{idx}]:{','.join(absent)}")
            blank_metadata = sorted(key for key in required_present if row.get(key) in {None, ""})
            if blank_metadata:
                empty_observed_metadata.append(f"{split}[{idx}]:{','.join(blank_metadata)}")
    status = "PASS"
    reasons: list[str] = []
    if not train or not evaluation:
        status = "FAIL"
        reasons.append("empty_split")
    if duplicate_train or duplicate_eval:
        status = "FAIL"
        reasons.append("duplicate_example_ids")
    if overlap:
        status = "FAIL"
        reasons.append("train_eval_id_overlap")
    if unseen_eval_labels:
        status = "FAIL"
        reasons.append("unseen_eval_labels")
    if missing_required_fields:
        status = "FAIL"
        reasons.append("missing_required_fields")
    result = {
        "status": status,
        "reasons": reasons,
        "n_train": len(train),
        "n_eval": len(evaluation),
        "train_label_counts": dict(sorted(train_labels.items())),
        "eval_label_counts": dict(sorted(eval_labels.items())),
        "duplicate_train_ids": duplicate_train[:20],
        "duplicate_eval_ids": duplicate_eval[:20],
        "train_eval_id_overlap": overlap[:20],
        "unseen_eval_labels": unseen_eval_labels,
        "missing_required_fields": missing_required_fields[:20],
        "empty_observed_metadata_count": len(empty_observed_metadata),
        "empty_observed_metadata_examples": empty_observed_metadata[:20],
    }
    if out_path is not None:
        write_json(out_path, result)
    return result


def validate_tool_policy_three_way(
    train_path: str | Path,
    calibration_path: str | Path,
    eval_path: str | Path,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate train, calibration, and final evaluation splits.

    The calibration split is used for model selection and confidence-threshold
    selection. The final evaluation split must remain disjoint from both train
    and calibration so reported performance and Monte Carlo scenarios are not
    tuned on the same examples.
    """
    train_cal = validate_tool_policy_splits(train_path, calibration_path)
    train_eval = validate_tool_policy_splits(train_path, eval_path)
    calibration = read_jsonl(calibration_path)
    evaluation = read_jsonl(eval_path)
    calibration_ids = {str(row.get("example_id", "")) for row in calibration} - {""}
    eval_ids = {str(row.get("example_id", "")) for row in evaluation} - {""}
    cal_eval_overlap = sorted(calibration_ids & eval_ids)
    calibration_labels = Counter(str(row.get("tool_label", "")) for row in calibration)
    eval_labels = Counter(str(row.get("tool_label", "")) for row in evaluation)
    missing_eval_labels_in_calibration = sorted(set(eval_labels) - set(calibration_labels))
    reasons = list(dict.fromkeys(train_cal["reasons"] + train_eval["reasons"]))
    if missing_eval_labels_in_calibration:
        reasons.append("calibration_missing_eval_labels")
    if cal_eval_overlap:
        reasons.append("calibration_eval_id_overlap")
    result = {
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "n_train": train_cal["n_train"],
        "n_calibration": len(calibration),
        "n_eval": len(evaluation),
        "train_label_counts": train_cal["train_label_counts"],
        "calibration_label_counts": dict(sorted(calibration_labels.items())),
        "eval_label_counts": dict(sorted(eval_labels.items())),
        "train_calibration_id_overlap": train_cal["train_eval_id_overlap"],
        "train_eval_id_overlap": train_eval["train_eval_id_overlap"],
        "calibration_eval_id_overlap": cal_eval_overlap[:20],
        "unseen_calibration_labels": train_cal["unseen_eval_labels"],
        "unseen_eval_labels": train_eval["unseen_eval_labels"],
        "missing_eval_labels_in_calibration": missing_eval_labels_in_calibration,
        "missing_required_fields": (
            train_cal["missing_required_fields"] + train_eval["missing_required_fields"]
        )[:40],
        "empty_observed_metadata_count": (
            train_cal["empty_observed_metadata_count"] + train_eval["empty_observed_metadata_count"]
        ),
    }
    if out_path is not None:
        write_json(out_path, result)
    return result
