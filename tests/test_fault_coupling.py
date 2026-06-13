from agentic_eval_framework.engine.execution_engine import (
    AgenticExecutionEngine,
    _synthetic_latency_ms,
)


def test_coupled_primary_tools_share_fault_and_latency_draw(tmp_path):
    profile = {
        "seed": 17,
        "timeout_prob": 0.25,
        "transient_error_prob": 0.25,
        "corrupt_evidence_prob": 0.25,
        "latency_base_ms": 5.0,
        "latency_jitter_ms": 20.0,
        "couple_across_tools": True,
    }
    scenario = {"scenario_id": "matched_case", "fault_profile": profile}
    engine = AgenticExecutionEngine(
        model_dir=tmp_path / "missing",
        db_path=tmp_path / "traces.sqlite",
    )
    try:
        assert engine._fault_type(scenario, "calendar_lookup", 0) == engine._fault_type(
            scenario, "search_places", 0
        )
        assert _synthetic_latency_ms(profile, "matched_case", "calendar_lookup", 0) == _synthetic_latency_ms(
            profile, "matched_case", "search_places", 0
        )
        # Terminal completion intentionally uses a separate stream.
        assert _synthetic_latency_ms(profile, "matched_case", "final_answer", 0) != _synthetic_latency_ms(
            profile, "matched_case", "calendar_lookup", 0
        )
    finally:
        engine.close()


def test_uncoupled_tools_keep_independent_latency_streams():
    profile = {"seed": 17, "latency_base_ms": 5.0, "latency_jitter_ms": 20.0}
    assert _synthetic_latency_ms(profile, "matched_case", "calendar_lookup", 0) != _synthetic_latency_ms(
        profile, "matched_case", "search_places", 0
    )
