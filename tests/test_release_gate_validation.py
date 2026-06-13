from __future__ import annotations

import pytest

from agentic_eval_framework.engine.release_gate import validate_release_thresholds


def test_release_threshold_validation_rejects_invalid_values():
    with pytest.raises(ValueError):
        validate_release_thresholds({"min_tool_accuracy": 1.2})
    with pytest.raises(ValueError):
        validate_release_thresholds({"unknown_threshold": 0.5})
