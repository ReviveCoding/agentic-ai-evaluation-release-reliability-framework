from agentic_eval_framework.simulation.monte_carlo import (
    MonteCarloConfig,
    _hybrid_predictions,
    build_monte_carlo_data,
    make_monte_carlo_scenarios,
)


def test_monte_carlo_generator_is_reproducible(tmp_path):
    cfg = MonteCarloConfig(
        seed=123,
        n_train_dialogues=30,
        n_dev_dialogues=16,
        replications=2,
        scenarios_per_replication=8,
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "processed"),
        model_dir=str(tmp_path / "model"),
        output_root=str(tmp_path / "out"),
    )
    build_monte_carlo_data(cfg)
    first = make_monte_carlo_scenarios(cfg)
    second = make_monte_carlo_scenarios(cfg)
    assert first == second
    assert len(first) == 16
    assert {s["difficulty"] for s in first} == {"easy", "medium", "hard", "stressed"}


def test_hybrid_uses_rule_when_metadata_is_consistent():
    reliable = {
        "observed_service": "Calendar_1",
        "observed_intent": "GetEvents",
        "observed_required_slots": ["date"],
        "observed_known_slots": {"date": "tomorrow"},
        "observed_missing_slots": [],
    }
    noisy = {
        "observed_service": "",
        "observed_intent": "UnknownIntent",
        "observed_required_slots": [],
        "observed_known_slots": {},
        "observed_missing_slots": [],
    }
    out = _hybrid_predictions([reliable, noisy], ["learned_a", "learned_b"], ["rule_a", "rule_b"])
    assert out == ["rule_a", "learned_b"]
