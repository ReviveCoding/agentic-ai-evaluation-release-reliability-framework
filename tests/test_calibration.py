import numpy as np

from agentic_eval_framework.models.calibration import expected_calibration_error, multiclass_brier_score


def test_calibration_metrics_are_zero_for_perfect_confident_predictions():
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    assert expected_calibration_error(probabilities, labels) == 0.0
    assert multiclass_brier_score(probabilities, labels) == 0.0


def test_calibration_metrics_increase_for_uncertain_predictions():
    probabilities = np.array([[0.9, 0.1], [0.9, 0.1]])
    labels = np.array([0, 1])
    assert expected_calibration_error(probabilities, labels) > 0.0
    assert multiclass_brier_score(probabilities, labels) > 0.0
