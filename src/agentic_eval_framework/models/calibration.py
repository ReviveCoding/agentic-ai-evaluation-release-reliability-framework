from __future__ import annotations

import numpy as np


def expected_calibration_error(probabilities: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Multiclass ECE using maximum predicted probability and correctness."""
    if len(y_true) == 0:
        return 0.0
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        left, right = edges[i], edges[i + 1]
        mask = (confidences >= left) & ((confidences < right) if i < n_bins - 1 else (confidences <= right))
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(correctness[mask].mean()) - float(confidences[mask].mean()))
    return float(ece)


def multiclass_brier_score(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    one_hot = np.zeros_like(probabilities, dtype=float)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def select_confidence_threshold(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    *,
    target_accuracy: float = 0.90,
    min_coverage: float = 0.50,
) -> dict[str, float]:
    """Select the lowest confidence threshold meeting a target selective accuracy.

    The threshold is chosen on the held-out evaluation split. If no threshold
    satisfies both conditions, the highest-confidence operating point is used.
    """
    if len(y_true) == 0:
        return {"threshold": 1.0, "coverage": 0.0, "selective_accuracy": 0.0}
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == y_true).astype(float)
    candidates = sorted(set([0.0, 1.0, *[float(x) for x in confidences]]))
    feasible: list[tuple[float, float, float]] = []
    all_points: list[tuple[float, float, float]] = []
    for threshold in candidates:
        mask = confidences >= threshold
        coverage = float(mask.mean())
        accuracy = float(correctness[mask].mean()) if np.any(mask) else 0.0
        point = (threshold, coverage, accuracy)
        all_points.append(point)
        if coverage >= min_coverage and accuracy >= target_accuracy:
            feasible.append(point)
    if feasible:
        threshold, coverage, accuracy = min(feasible, key=lambda x: x[0])
    else:
        threshold, coverage, accuracy = max(all_points, key=lambda x: (x[2], x[1], -x[0]))
    return {
        "threshold": float(threshold),
        "coverage": float(coverage),
        "selective_accuracy": float(accuracy),
        "target_accuracy": float(target_accuracy),
        "min_coverage": float(min_coverage),
    }
