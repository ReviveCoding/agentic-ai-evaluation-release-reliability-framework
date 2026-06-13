from agentic_eval_framework.engine.replay import _compare_replay


def _run(confidence: float, latency: float):
    return {
        "model_fingerprint": "same",
        "predicted_tools": ["search_docs", "final_answer"],
        "release_decision": "REVIEW",
        "failure_type": "trajectory_failure",
        "completion_suppressed": False,
        "scores": {"tool_accuracy": 1.0, "model_confidence": confidence, "max_step_latency_ms": latency},
        "steps": [
            {
                "tool_name": "search_docs",
                "execution": {"status": "success", "final_error": False, "retry_count": 0, "recovered_after_retry": False},
                "observation": {"evidence_id": "intent::Services_1::Search", "retrieved_doc_ids": ["intent::Services_1::Search"], "retrieval_target_doc_id": "intent::Services_1::Search", "evidence_compatible": True, "grounded": True},
            }
        ],
    }


def test_replay_ignores_latency_and_tolerates_small_float_drift():
    original = _run(0.9000000, 10.0)
    replayed = _run(0.9000005, 99.0)
    replayed["replay_mode"] = "full_policy_reexecution"
    result = _compare_replay(original, replayed)
    assert result["verified"] is True


def test_replay_detects_retrieval_order_change():
    original = _run(0.9, 10.0)
    replayed = _run(0.9, 10.0)
    replayed["replay_mode"] = "execution_only"
    replayed["steps"][0]["observation"]["retrieved_doc_ids"] = ["wrong"]
    result = _compare_replay(original, replayed)
    assert result["verified"] is False
    assert result["checks"]["stable_step_signature_match"] is False
