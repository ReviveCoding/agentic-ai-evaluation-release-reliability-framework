import numpy as np

from agentic_eval_framework.simulation.monte_carlo import _render_utterance


def test_calibration_and_test_use_distinct_held_out_templates():
    spec = {"templates": ["train one", "calibration only", "test only"]}
    rng = np.random.default_rng(7)
    train_outputs = {_render_utterance(spec, {}, rng, "train") for _ in range(30)}
    calibration_outputs = {_render_utterance(spec, {}, rng, "dev") for _ in range(10)}
    test_outputs = {_render_utterance(spec, {}, rng, "test") for _ in range(10)}
    assert train_outputs == {"train one"}
    assert calibration_outputs == {"calibration only"}
    assert test_outputs == {"test only"}
