import numpy as np
import pytest

from agentic_eval_framework.models.calibration import select_confidence_threshold


def reference_selection(probabilities, labels, target_accuracy, min_coverage):
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == labels).astype(float)
    points = []
    for threshold in sorted({0.0, 1.0, *map(float, confidence)}):
        mask = confidence >= threshold
        points.append(
            (
                threshold,
                float(mask.mean()),
                float(correct[mask].mean()) if mask.any() else 0.0,
            )
        )
    feasible = [
        point
        for point in points
        if point[1] >= min_coverage and point[2] >= target_accuracy
    ]
    return (
        min(feasible, key=lambda point: point[0])
        if feasible
        else max(points, key=lambda point: (point[2], point[1], -point[0]))
    )


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("target,coverage", [(0.0, 0.0), (0.9, 0.5), (1.0, 1.0)])
@pytest.mark.parametrize("tied", [False, True])
def test_threshold_selection_matches_mask_reference(seed, target, coverage, tied):
    rng = np.random.default_rng(seed)
    probabilities = rng.dirichlet([1.0, 1.0, 1.0], size=73)
    if tied:
        probabilities = np.round(probabilities, 1)
    labels = rng.integers(0, 3, size=73)
    expected = reference_selection(probabilities, labels, target, coverage)
    actual = select_confidence_threshold(
        probabilities, labels, target_accuracy=target, min_coverage=coverage
    )
    assert (
        actual["threshold"],
        actual["coverage"],
        actual["selective_accuracy"],
    ) == expected


def test_empty_threshold_selection_keeps_existing_result():
    assert select_confidence_threshold(np.empty((0, 3)), np.array([], dtype=int)) == {
        "threshold": 1.0,
        "coverage": 0.0,
        "selective_accuracy": 0.0,
    }


def test_nonfinite_confidence_keeps_mask_comparison_behavior():
    probabilities = np.array([[np.inf, 0.1], [0.3, 0.7], [-np.inf, -np.inf]])
    labels = np.array([0, 0, 1])
    expected = reference_selection(probabilities, labels, 0.9, 0.5)
    actual = select_confidence_threshold(probabilities, labels)
    assert (
        actual["threshold"],
        actual["coverage"],
        actual["selective_accuracy"],
    ) == expected
