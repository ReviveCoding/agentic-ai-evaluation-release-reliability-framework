import numpy as np

from agentic_eval_framework.simulation.monte_carlo import SERVICE_SPECS, _render_utterance


def test_train_calibration_and_test_use_distinct_templates():
    spec = SERVICE_SPECS[0]
    values = {"location": ["Boston"], "wifi": ["free"]}
    train = _render_utterance(spec, values, np.random.default_rng(1), "train")
    calibration = _render_utterance(spec, values, np.random.default_rng(1), "dev")
    test = _render_utterance(spec, values, np.random.default_rng(1), "test")
    assert len({train, calibration, test}) == 3
