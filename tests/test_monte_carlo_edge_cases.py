from __future__ import annotations

from collections import Counter

from agentic_eval_framework.simulation.monte_carlo import (
    MonteCarloConfig,
    SERVICE_SPECS,
    build_monte_carlo_data,
    make_monte_carlo_scenarios,
)
from agentic_eval_framework.utils.io import read_jsonl


def test_monte_carlo_generates_exact_non_divisible_count(tmp_path):
    cfg = MonteCarloConfig(
        seed=77,
        n_train_dialogues=20,
        n_dev_dialogues=14,
        replications=3,
        scenarios_per_replication=10,
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "processed"),
        model_dir=str(tmp_path / "model"),
        output_root=str(tmp_path / "out"),
    )
    build_monte_carlo_data(cfg)
    scenarios = make_monte_carlo_scenarios(cfg)
    assert len(scenarios) == 30
    by_rep = Counter(s["replication_id"] for s in scenarios)
    assert set(by_rep.values()) == {10}
    for rep in by_rep:
        counts = Counter(s["difficulty"] for s in scenarios if s["replication_id"] == rep)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_small_synthetic_splits_cover_all_intents_when_large_enough(tmp_path):
    cfg = MonteCarloConfig(
        seed=88,
        n_train_dialogues=len(SERVICE_SPECS),
        n_dev_dialogues=len(SERVICE_SPECS),
        replications=1,
        scenarios_per_replication=4,
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "processed"),
        model_dir=str(tmp_path / "model"),
        output_root=str(tmp_path / "out"),
    )
    info = build_monte_carlo_data(cfg)
    train = read_jsonl(info["train_path"])
    evaluation = read_jsonl(info["eval_path"])
    expected = {(s["service"], s["intent"]) for s in SERVICE_SPECS}
    assert {(r["service"], r["intent"]) for r in train} >= expected
    assert {(r["service"], r["intent"]) for r in evaluation} >= expected
