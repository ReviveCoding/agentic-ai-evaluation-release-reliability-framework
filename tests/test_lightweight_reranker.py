from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from agentic_eval_framework.retrieval.top1_reranker import FEATURE_NAMES, LightweightIntentReranker


def test_lightweight_reranker_returns_positive_probability():
    x = np.asarray([[0.0] * len(FEATURE_NAMES), [1.0] * len(FEATURE_NAMES)])
    y = np.asarray([0, 1])
    model = LogisticRegression(solver="liblinear", random_state=42).fit(x, y)
    reranker = LightweightIntentReranker(model, FEATURE_NAMES)
    scores = reranker.score(x.tolist())
    assert len(scores) == 2
    assert scores[1] > scores[0]
