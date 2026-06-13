from __future__ import annotations

from pathlib import Path

from agentic_eval_framework.simulation.monte_carlo import (
    MonteCarloConfig,
    _stratified_panel_indexes,
    build_monte_carlo_data,
    make_monte_carlo_scenarios,
)
from agentic_eval_framework.utils.io import read_jsonl


def test_stratified_panel_includes_critical_tools(tmp_path: Path):
    cfg = MonteCarloConfig(
        seed=123,
        n_train_dialogues=80,
        n_dev_dialogues=80,
        replications=1,
        scenarios_per_replication=40,
        output_root=str(tmp_path / "out"),
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "processed"),
        model_dir=str(tmp_path / "model"),
    )
    build_monte_carlo_data(cfg)
    base = read_jsonl(Path(cfg.processed_dir) / "golden_trajectories.jsonl")
    import numpy as np
    indexes = _stratified_panel_indexes(base, 10, np.random.default_rng(123))
    tools = {(base[idx].get("expected_tools") or [""])[0] for idx in indexes}
    assert {"safety_check", "calendar_write", "ask_clarification"}.issubset(tools)

    scenarios = make_monte_carlo_scenarios(cfg)
    first_tools = [(scenario.get("expected_tools") or [""])[0] for scenario in scenarios]
    assert first_tools.count("safety_check") >= 4
    assert first_tools.count("calendar_write") >= 4
    assert first_tools.count("ask_clarification") >= 4
