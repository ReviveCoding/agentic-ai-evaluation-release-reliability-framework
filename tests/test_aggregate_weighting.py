from agentic_eval_framework.engine.execution_engine import aggregate_runs


def _run(run_id, stats, scores):
    return {
        "run_id": run_id,
        "release_decision": "PASS",
        "scores": scores,
        "execution_stats": stats,
        "steps": [{"latency_ms": 1.0}],
        "simulated_latency_ms": 1.0,
    }


def test_retry_recovery_uses_ratio_of_sums_not_mean_of_run_rates():
    runs = [
        _run(
            "no_retry",
            {"total_steps": 10, "total_attempts": 10, "total_timeouts": 0, "retried_steps": 0, "recovered_steps": 0, "failed_steps": 0},
            {"retry_recovery_rate": 1.0, "retry_rate": 0.0, "execution_error_rate": 0.0, "timeout_rate": 0.0},
        ),
        _run(
            "failed_retry",
            {"total_steps": 1, "total_attempts": 2, "total_timeouts": 1, "retried_steps": 1, "recovered_steps": 0, "failed_steps": 1},
            {"retry_recovery_rate": 0.0, "retry_rate": 1.0, "execution_error_rate": 1.0, "timeout_rate": 0.5},
        ),
    ]
    agg = aggregate_runs(runs)
    assert agg["retry_recovery_rate"] == 0.0
    assert agg["retry_recovery_coverage"] == 1 / 11
    assert agg["retry_rate"] == 1 / 11
    assert agg["execution_error_rate"] == 1 / 11
    assert agg["timeout_rate"] == 1 / 12
