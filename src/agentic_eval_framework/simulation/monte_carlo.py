from __future__ import annotations

import asyncio
import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agentic_eval_framework.data.build_golden_trajectories import build_golden_trajectories
from agentic_eval_framework.data.build_service_docs import build_service_docs
from agentic_eval_framework.data.build_tool_policy_dataset import build_tool_policy_dataset
from agentic_eval_framework.data.validate_dataset import validate_tool_policy_three_way
from agentic_eval_framework.engine.execution_engine import AgenticExecutionEngine, aggregate_runs, scenario_to_model_row
from agentic_eval_framework.models.rule_policy import RulePolicy
from agentic_eval_framework.models.train_tool_policy import train_tool_policy
from agentic_eval_framework.storage.sqlite_store import SQLiteTraceStore
from agentic_eval_framework.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl


SERVICE_SPECS: list[dict[str, Any]] = [
    {
        "service": "Hotels_1", "description": "Find lodging options by destination and amenities.",
        "intent": "SearchHotel", "intent_description": "Find lodging matching user constraints.",
        "required": ["location"], "optional": ["wifi", "price"], "weight": 0.18,
        "templates": [
            "Find a hotel in {location} with {wifi} wifi.",
            "I need lodging around {location}; wifi preference is {wifi}.",
            "Show hotel options for {location}.",
            "Could you look up places to stay in {location}?",
        ],
    },
    {
        "service": "Calendar_1", "description": "Retrieve or create calendar events.",
        "intent": "GetEvents", "intent_description": "Retrieve scheduled events for a date.",
        "required": ["date"], "optional": [], "weight": 0.13,
        "templates": [
            "What meetings do I have on {date}?",
            "Show my calendar for {date}.",
            "Do I have anything scheduled {date}?",
        ],
    },
    {
        "service": "Calendar_1", "description": "Retrieve or create calendar events.",
        "intent": "CreateEvent", "intent_description": "Create a calendar event.",
        "required": ["date", "time", "title"], "optional": [], "weight": 0.13,
        "templates": [
            "Schedule {title} on {date} at {time}.",
            "Put {title} on my calendar for {date}, {time}.",
            "Create an event called {title}.",
        ],
    },
    {
        "service": "Weather_1", "description": "Retrieve weather forecasts by place and date.",
        "intent": "GetWeather", "intent_description": "Get a weather forecast.",
        "required": ["location"], "optional": ["date"], "weight": 0.15,
        "templates": [
            "What is the weather in {location} on {date}?",
            "Will it rain around {location} {date}?",
            "Check the forecast for {location}.",
        ],
    },
    {
        "service": "Media_1", "description": "Search a catalog of movies and music.",
        "intent": "SearchMedia", "intent_description": "Search media by genre or title.",
        "required": [], "optional": ["genre", "title"], "weight": 0.13,
        "templates": [
            "Find a {genre} movie.",
            "Search for media called {title}.",
            "Recommend something in the {genre} category.",
        ],
    },
    {
        "service": "Documents_1", "description": "Search personal or enterprise documents by topic.",
        "intent": "SearchDocument", "intent_description": "Retrieve documents relevant to a topic.",
        "required": ["topic"], "optional": [], "weight": 0.16,
        "templates": [
            "Find the document about {topic}.",
            "Search my files for {topic}.",
            "Locate notes related to {topic}.",
        ],
    },
    {
        "service": "Bank_1", "description": "Handle sensitive banking and money transfer requests.",
        "intent": "TransferMoney", "intent_description": "Transfer money between accounts.",
        "required": ["account", "amount"], "optional": [], "weight": 0.12,
        "templates": [
            "Transfer {amount} dollars to my {account} account.",
            "Move {amount} to {account}.",
            "Send money to my {account} account.",
        ],
    },
]

SLOT_VALUES = {
    "location": ["New York", "Cambridge", "Seattle", "Austin", "Boston", "Chicago"],
    "wifi": ["free", "reliable", "no preference for"],
    "price": ["budget", "midrange", "premium"],
    "date": ["tomorrow", "next Monday", "this weekend", "June 20", "Friday"],
    "time": ["9 AM", "noon", "3 PM", "6:30 PM"],
    "title": ["project review", "lunch with Dana", "doctor appointment", "team sync"],
    "genre": ["science fiction", "comedy", "documentary", "jazz"],
    "topic": ["expense policy", "project Alpha", "travel reimbursement", "quarterly planning"],
    "account": ["savings", "checking", "brokerage"],
    "amount": ["50", "200", "500", "1200"],
}

DIFFICULTY = {
    "easy": {"weight": 0.35, "service_noise": 0.01, "intent_noise": 0.01, "slot_metadata_noise": 0.02, "utterance_noise": 0.04, "fault": (0.005, 0.010, 0.005), "latency": (5.0, 8.0)},
    "medium": {"weight": 0.35, "service_noise": 0.08, "intent_noise": 0.08, "slot_metadata_noise": 0.10, "utterance_noise": 0.15, "fault": (0.020, 0.050, 0.025), "latency": (8.0, 15.0)},
    "hard": {"weight": 0.20, "service_noise": 0.22, "intent_noise": 0.22, "slot_metadata_noise": 0.28, "utterance_noise": 0.32, "fault": (0.050, 0.100, 0.060), "latency": (12.0, 28.0)},
    "stressed": {"weight": 0.10, "service_noise": 0.40, "intent_noise": 0.40, "slot_metadata_noise": 0.45, "utterance_noise": 0.50, "fault": (0.100, 0.180, 0.120), "latency": (20.0, 60.0)},
}


@dataclass
class MonteCarloConfig:
    seed: int = 20260611
    n_train_dialogues: int = 900
    n_dev_dialogues: int = 300
    n_test_dialogues: int | None = None
    replications: int = 12
    scenarios_per_replication: int = 100
    concurrency: int = 32
    max_retries: int = 2
    training_backend: str = "sklearn"
    model_name: str = "distilbert-base-uncased"
    epochs: int = 2
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    output_root: str = "outputs/monte_carlo"
    raw_dir: str = "data/monte_carlo_raw/sgd"
    processed_dir: str = "data/monte_carlo_processed"
    model_dir: str = "models/monte_carlo_tool_policy"

    def __post_init__(self) -> None:
        if self.n_test_dialogues is None:
            self.n_test_dialogues = self.n_dev_dialogues
        for name in (
            "n_train_dialogues", "n_dev_dialogues", "n_test_dialogues",
            "replications", "scenarios_per_replication", "concurrency"
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.training_backend not in {"sklearn", "transformer"}:
            raise ValueError("training_backend must be sklearn or transformer")


def _schema_payload() -> list[dict[str, Any]]:
    by_service: dict[str, dict[str, Any]] = {}
    for spec in SERVICE_SPECS:
        service = spec["service"]
        schema = by_service.setdefault(service, {
            "service_name": service,
            "description": spec["description"],
            "slots": [],
            "intents": [],
        })
        slot_names = set(spec["required"] + spec["optional"])
        existing = {s["name"] for s in schema["slots"]}
        for slot in sorted(slot_names - existing):
            schema["slots"].append({"name": slot, "description": f"Value for {slot}"})
        schema["intents"].append({
            "name": spec["intent"],
            "description": spec["intent_description"],
            "required_slots": spec["required"],
            "optional_slots": spec["optional"],
        })
    return list(by_service.values())


def _choose_spec(rng: np.random.Generator) -> dict[str, Any]:
    weights = np.array([s["weight"] for s in SERVICE_SPECS], dtype=float)
    weights /= weights.sum()
    return SERVICE_SPECS[int(rng.choice(len(SERVICE_SPECS), p=weights))]


def _fill_slots(spec: dict[str, Any], rng: np.random.Generator, missing_prob: float) -> tuple[dict[str, list[str]], list[str]]:
    values: dict[str, list[str]] = {}
    missing: list[str] = []
    for slot in spec["required"]:
        if rng.random() < missing_prob:
            missing.append(slot)
        else:
            values[slot] = [str(rng.choice(SLOT_VALUES[slot]))]
    for slot in spec["optional"]:
        if rng.random() < 0.55:
            values[slot] = [str(rng.choice(SLOT_VALUES[slot]))]
    return values, missing


def _render_utterance(
    spec: dict[str, Any], values: dict[str, list[str]], rng: np.random.Generator, split: str
) -> str:
    templates = list(spec["templates"])
    # Three-way paraphrase holdout prevents model/threshold selection on the
    # same surface form used for final Monte Carlo evaluation.
    if len(templates) >= 3:
        if split == "train":
            pool = templates[:-2]
        elif split == "dev":
            pool = templates[-2:-1]
        else:  # held-out test
            pool = templates[-1:]
    elif len(templates) == 2:
        pool = templates[:1] if split == "train" else templates[1:]
    else:
        pool = templates
    template = str(rng.choice(pool))
    fallback = {
        "location": "somewhere nearby", "wifi": "any", "price": "any",
        "date": "sometime", "time": "a suitable time", "title": "an event",
        "genre": "interesting", "topic": "that topic", "account": "another",
        "amount": "some",
    }
    rendered = template
    for slot, default in fallback.items():
        rendered = rendered.replace("{" + slot + "}", str(values.get(slot, [default])[0]))
    return rendered


def _make_dialogue(
    idx: int, rng: np.random.Generator, split: str, spec: dict[str, Any] | None = None,
    force_complete: bool = False,
) -> dict[str, Any]:
    spec = spec or _choose_spec(rng)
    missing_prob = 0.0 if force_complete else (0.22 if split == "train" else 0.28)
    slot_values, _ = _fill_slots(spec, rng, missing_prob)
    utterance = _render_utterance(spec, slot_values, rng, split)
    return {
        "dialogue_id": f"mc_{split}_{idx:06d}",
        "services": [spec["service"]],
        "turns": [
            {
                "speaker": "USER",
                "utterance": utterance,
                "frames": [{
                    "service": spec["service"],
                    "state": {"active_intent": spec["intent"], "slot_values": slot_values},
                }],
            },
            {"speaker": "SYSTEM", "utterance": "Synthetic response placeholder.", "frames": []},
        ],
    }


def generate_raw_dataset(config: MonteCarloConfig) -> dict[str, int]:
    rng = np.random.default_rng(config.seed)
    schemas = _schema_payload()
    raw = Path(config.raw_dir)
    counts = {
        "train": config.n_train_dialogues,
        "dev": config.n_dev_dialogues,
        "test": int(config.n_test_dialogues),
    }
    for split, n in counts.items():
        split_dir = ensure_dir(raw / split)
        write_json(split_dir / "schema.json", schemas)
        # Guarantee intent/tool coverage for small validation splits, then sample
        # the remainder using the production-like class weights.
        specs: list[tuple[dict[str, Any], bool]] = []
        specs.extend((copy.deepcopy(spec), True) for spec in SERVICE_SPECS[: min(n, len(SERVICE_SPECS))])
        while len(specs) < n:
            specs.append((copy.deepcopy(_choose_spec(rng)), False))
        order = rng.permutation(len(specs))
        dialogues = [
            _make_dialogue(i, rng, split, specs[int(j)][0], force_complete=specs[int(j)][1])
            for i, j in enumerate(order)
        ]
        write_json(split_dir / "dialogues_001.json", dialogues)
    return counts


def build_monte_carlo_data(config: MonteCarloConfig) -> dict[str, Any]:
    generate_raw_dataset(config)
    processed = ensure_dir(config.processed_dir)
    train_path = processed / "tool_policy_train.jsonl"
    calibration_path = processed / "tool_policy_calibration.jsonl"
    eval_path = processed / "tool_policy_eval.jsonl"
    golden_path = processed / "golden_trajectories.jsonl"
    service_docs_path = processed / "service_docs.jsonl"
    train_rows = build_tool_policy_dataset(config.raw_dir, "train", train_path)
    calibration_rows = build_tool_policy_dataset(config.raw_dir, "dev", calibration_path)
    eval_rows = build_tool_policy_dataset(config.raw_dir, "test", eval_path)
    # Add controlled metadata-dropout copies so the learned policy can fall back
    # to utterance/schema cues when observed service metadata is incomplete.
    aug_rng = np.random.default_rng(config.seed + 77)
    augmented: list[dict[str, Any]] = []
    for row in train_rows:
        if aug_rng.random() < 0.70:
            aug = copy.deepcopy(row)
            aug["example_id"] = f"{row['example_id']}_aug"
            if aug_rng.random() < 0.55:
                aug["service"] = ""
                aug["service_description"] = ""
            if aug_rng.random() < 0.45:
                aug["intent"] = "UnknownIntent"
            if aug_rng.random() < 0.35 and aug.get("missing_slots"):
                aug["missing_slots"] = []
            augmented.append(aug)
    train_rows = train_rows + augmented
    write_jsonl(train_path, train_rows)
    build_service_docs(config.raw_dir, ("train", "dev", "test"), service_docs_path)
    build_golden_trajectories(eval_path, golden_path)
    validation = validate_tool_policy_three_way(train_path, calibration_path, eval_path)
    if validation["status"] != "PASS":
        raise ValueError(f"Monte Carlo dataset validation failed: {validation['reasons']}")
    return {
        "train_path": str(train_path),
        "calibration_path": str(calibration_path),
        "eval_path": str(eval_path),
        "golden_path": str(golden_path), "service_docs_path": str(service_docs_path),
        "n_train_rows": len(train_rows),
        "n_calibration_rows": len(calibration_rows),
        "n_eval_rows": len(eval_rows),
        "validation": validation,
    }


def _perturb_text(text: str, rng: np.random.Generator, probability: float) -> str:
    if rng.random() >= probability:
        return text
    fillers = ["please", "actually", "quick question", "when you can", "for me"]
    words = text.split()
    if len(words) > 4 and rng.random() < 0.6:
        drop_idx = int(rng.integers(1, len(words) - 1))
        words.pop(drop_idx)
    if rng.random() < 0.5:
        words.insert(0, str(rng.choice(fillers)))
    if words and rng.random() < 0.35:
        idx = int(rng.integers(0, len(words)))
        word = words[idx]
        if len(word) > 4:
            pos = int(rng.integers(1, len(word) - 1))
            words[idx] = word[:pos] + word[pos + 1:]
    return " ".join(words)


def _wrong_service(true_service: str, rng: np.random.Generator) -> str:
    choices = sorted({s["service"] for s in SERVICE_SPECS if s["service"] != true_service})
    return str(rng.choice(choices))


def _service_description(service: str) -> str:
    for spec in SERVICE_SPECS:
        if spec["service"] == service:
            return spec["description"]
    return ""


def _required_slots(service: str, intent: str) -> list[str]:
    for spec in SERVICE_SPECS:
        if spec["service"] == service and spec["intent"] == intent:
            return list(spec["required"])
    return []




def _stratified_panel_indexes(
    base: list[dict[str, Any]], panel_size: int, rng: np.random.Generator
) -> list[int]:
    """Select a matched panel with minimum critical tool coverage.

    Weighted random sampling alone can omit rare safety/write/clarification
    classes from an entire replication. We seed the panel with one example per
    available first-tool class, prioritizing critical decision surfaces, then
    fill the remainder randomly to preserve natural variation.
    """
    groups: dict[str, list[int]] = {}
    for idx, scenario in enumerate(base):
        first = str((scenario.get("expected_tools") or ["unknown"])[0])
        groups.setdefault(first, []).append(idx)
    critical = ["safety_check", "calendar_write", "ask_clarification"]
    ordered_tools = [tool for tool in critical if tool in groups] + [
        tool for tool in sorted(groups) if tool not in critical
    ]
    selected: list[int] = []
    for tool in ordered_tools:
        if len(selected) >= panel_size:
            break
        selected.append(int(rng.choice(groups[tool])))
    while len(selected) < panel_size:
        selected.append(int(rng.integers(0, len(base))))
    order = rng.permutation(len(selected))
    return [selected[int(i)] for i in order]

def make_monte_carlo_scenarios(config: MonteCarloConfig) -> list[dict[str, Any]]:
    base = read_jsonl(Path(config.processed_dir) / "golden_trajectories.jsonl")
    if not base:
        raise ValueError("No golden trajectories were generated")
    scenarios: list[dict[str, Any]] = []
    difficulty_names = list(DIFFICULTY)
    for rep in range(config.replications):
        rng = np.random.default_rng(config.seed + 1009 * rep)
        # Matched-panel design: the same base tasks are exposed to every
        # difficulty level. This prevents service/intent mix from masquerading
        # as a difficulty effect in Monte Carlo slice comparisons.
        panel_size = max(1, math.ceil(config.scenarios_per_replication / len(difficulty_names)))
        base_indexes = _stratified_panel_indexes(base, panel_size, rng)
        produced = 0
        for panel_pos, base_idx in enumerate(base_indexes):
            for difficulty in difficulty_names:
                if produced >= config.scenarios_per_replication:
                    break
                scenario = copy.deepcopy(base[int(base_idx)])
                params = DIFFICULTY[difficulty]
                true_service = scenario.get("service", "")
                true_intent = scenario.get("intent", "")
                observed_service = true_service
                if rng.random() < params["service_noise"]:
                    observed_service = "" if rng.random() < 0.45 else _wrong_service(true_service, rng)
                observed_intent = true_intent
                if rng.random() < params["intent_noise"]:
                    observed_intent = "UnknownIntent" if rng.random() < 0.5 else str(rng.choice(all_intents := [x["intent"] for x in SERVICE_SPECS]))
                observed_missing = list(scenario.get("missing_slots", []))
                observed_known = dict(scenario.get("known_slots", {}))
                observed_risk = list(scenario.get("risk_flags", []))
                observed_required = _required_slots(observed_service, observed_intent)
                if rng.random() < params["slot_metadata_noise"]:
                    if observed_missing and rng.random() < 0.7:
                        observed_missing = []
                    elif observed_known:
                        key = str(rng.choice(list(observed_known)))
                        observed_known.pop(key, None)
                    if observed_risk and rng.random() < 0.5:
                        observed_risk = []
                timeout_prob, transient_prob, corrupt_prob = params["fault"]
                latency_base, latency_jitter = params["latency"]
                scenario.update({
                    "scenario_id": f"MC_R{rep:03d}_P{panel_pos:04d}_{difficulty}",
                    "replication_id": rep,
                    "panel_id": panel_pos,
                    "difficulty": difficulty,
                    "simulation_seed": config.seed + 1009 * rep,
                    "user_request": _perturb_text(scenario.get("user_request", ""), rng, params["utterance_noise"]),
                    "observed_service": observed_service,
                    "observed_intent": observed_intent,
                    "observed_service_description": _service_description(observed_service),
                    "observed_known_slots": observed_known,
                    "observed_missing_slots": observed_missing,
                    "observed_required_slots": observed_required,
                    "observed_risk_flags": observed_risk,
                    "fault_profile": {
                        "seed": config.seed + 1009 * rep,
                        "timeout_prob": timeout_prob,
                        "transient_error_prob": transient_prob,
                        "corrupt_evidence_prob": corrupt_prob,
                        "latency_base_ms": latency_base,
                        "latency_jitter_ms": latency_jitter,
                        "couple_across_tools": True,
                    },
                })
                scenarios.append(scenario)
                produced += 1
        # Handle non-divisible requested sizes without changing the matched core.
        while produced < config.scenarios_per_replication:
            scenario = copy.deepcopy(base[int(rng.integers(0, len(base)))])
            difficulty = difficulty_names[produced % len(difficulty_names)]
            params = DIFFICULTY[difficulty]
            scenario.update({
                "scenario_id": f"MC_R{rep:03d}_EXTRA_{produced:04d}_{difficulty}",
                "replication_id": rep, "panel_id": -1, "difficulty": difficulty,
                "simulation_seed": config.seed + 1009 * rep,
                "observed_service": scenario.get("service", ""),
                "observed_intent": scenario.get("intent", ""),
                "observed_service_description": _service_description(scenario.get("service", "")),
                "observed_known_slots": scenario.get("known_slots", {}),
                "observed_missing_slots": scenario.get("missing_slots", []),
                "observed_required_slots": scenario.get("required_slots", []),
                "observed_risk_flags": scenario.get("risk_flags", []),
                "fault_profile": {
                    "seed": config.seed + 1009 * rep,
                    "timeout_prob": params["fault"][0], "transient_error_prob": params["fault"][1],
                    "corrupt_evidence_prob": params["fault"][2],
                    "latency_base_ms": params["latency"][0], "latency_jitter_ms": params["latency"][1],
                    "couple_across_tools": True,
                },
            })
            scenarios.append(scenario)
            produced += 1
    expected = config.replications * config.scenarios_per_replication
    if len(scenarios) != expected:
        raise AssertionError(f"generated {len(scenarios)} scenarios, expected {expected}")
    return scenarios


def _bootstrap_ci(
    values: list[float], seed: int, n_bootstrap: int = 2000, *, bounded: bool = True
) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std": 0.0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if arr.size == 1:
        low = high = mean
    else:
        rng = np.random.default_rng(seed)
        samples = rng.choice(arr, size=(n_bootstrap, arr.size), replace=True).mean(axis=1)
        low, high = (float(x) for x in np.quantile(samples, [0.025, 0.975]))
    if bounded:
        low, high = max(0.0, low), min(1.0, high)
    return {"mean": mean, "ci_low": low, "ci_high": high, "std": std, "method": "percentile_bootstrap"}


def _replication_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in runs:
        key = (run["variant"], int(run["scenario"].get("replication_id", 0)))
        groups.setdefault(key, []).append(run)
    out = []
    for (variant, rep), group in sorted(groups.items()):
        agg = aggregate_runs(group)
        out.append({"variant": variant, "replication_id": rep, **agg})
    return out


def _variant_summary(runs: list[dict[str, Any]], rep_summaries: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in sorted({r["variant"] for r in runs}):
        vruns = [r for r in runs if r["variant"] == variant]
        vreps = [r for r in rep_summaries if r["variant"] == variant]
        aggregate = aggregate_runs(vruns)
        cis: dict[str, Any] = {}
        for metric_idx, metric in enumerate(["tool_accuracy", "trajectory_success", "groundedness", "clarification_recall", "safety_block_rate", "retry_recovery_rate", "execution_error_rate", "timeout_rate", "model_confidence"]):
            cis[metric] = _bootstrap_ci(
                [float(rep.get(metric, 0.0)) for rep in vreps],
                seed + 97 * (metric_idx + 1) + sum(ord(c) for c in variant),
            )
        pass_rates = []
        for rep in vreps:
            decisions = rep.get("release_decisions", {})
            pass_rates.append(float(decisions.get("PASS", 0)) / max(1, int(rep.get("num_runs", 0))))
        cis["pass_rate"] = _bootstrap_ci(pass_rates, seed + 7919 + sum(ord(c) for c in variant))
        by_difficulty = {}
        for difficulty in DIFFICULTY:
            druns = [r for r in vruns if r["scenario"].get("difficulty") == difficulty]
            by_difficulty[difficulty] = aggregate_runs(druns)
        by_tool = {}
        tool_names = sorted({str((r.get("expected_tools") or ["unknown"])[0]) for r in vruns})
        for tool_name in tool_names:
            truns = [r for r in vruns if str((r.get("expected_tools") or ["unknown"])[0]) == tool_name]
            by_tool[tool_name] = aggregate_runs(truns)
        result[variant] = {
            "aggregate": aggregate, "confidence_intervals": cis,
            "by_difficulty": by_difficulty, "by_tool": by_tool,
        }
    return result


def _paired_comparisons(rep_summaries: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    by_variant_rep = {(r["variant"], int(r["replication_id"])): r for r in rep_summaries}
    comparisons: dict[str, Any] = {}
    metrics = ["tool_accuracy", "trajectory_success", "groundedness"]
    for baseline in ["rule", "learned", "hybrid_no_retry"]:
        key = f"hybrid_minus_{baseline}"
        comparisons[key] = {}
        reps = sorted({rep for variant, rep in by_variant_rep if variant == "hybrid"} & {rep for variant, rep in by_variant_rep if variant == baseline})
        for metric_idx, metric in enumerate(metrics):
            diffs = [
                float(by_variant_rep[("hybrid", rep)].get(metric, 0.0))
                - float(by_variant_rep[(baseline, rep)].get(metric, 0.0))
                for rep in reps
            ]
            comparisons[key][metric] = _bootstrap_ci(
                diffs, seed + 1543 * (metric_idx + 1) + sum(ord(c) for c in baseline), bounded=False
            )
    return comparisons


def _sanity_checks(summary: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for variant, payload in summary.items():
        diff = payload["by_difficulty"]
        easy = diff.get("easy", {}).get("trajectory_success", 0.0)
        stressed = diff.get("stressed", {}).get("trajectory_success", 0.0)
        checks.append({"name": f"{variant}_stress_degrades_or_matches", "passed": easy + 0.02 >= stressed, "values": {"easy": easy, "stressed": stressed}})
        checks.append({"name": f"{variant}_trace_complete", "passed": payload["aggregate"].get("trace_completeness", 0.0) >= 0.99, "value": payload["aggregate"].get("trace_completeness", 0.0)})
    if "hybrid" in summary and "rule" in summary and "learned" in summary:
        hybrid = summary["hybrid"]["aggregate"].get("tool_accuracy", 0.0)
        best_single = max(summary["rule"]["aggregate"].get("tool_accuracy", 0.0), summary["learned"]["aggregate"].get("tool_accuracy", 0.0))
        checks.append({"name": "hybrid_matches_best_single_policy", "passed": hybrid + 0.01 >= best_single, "values": {"hybrid": hybrid, "best_single": best_single}})
    if "hybrid" in summary and "hybrid_no_retry" in summary:
        with_retry = summary["hybrid"]["aggregate"]
        no_retry = summary["hybrid_no_retry"]["aggregate"]
        checks.append({"name": "retry_reduces_execution_errors", "passed": with_retry.get("execution_error_rate", 1.0) <= no_retry.get("execution_error_rate", 1.0), "values": {"with_retry": with_retry.get("execution_error_rate"), "without_retry": no_retry.get("execution_error_rate")}})
        checks.append({"name": "retry_improves_or_preserves_groundedness", "passed": with_retry.get("groundedness", 0.0) + 1e-9 >= no_retry.get("groundedness", 0.0), "values": {"with_retry": with_retry.get("groundedness"), "without_retry": no_retry.get("groundedness")}})
        if int(with_retry.get("num_runs", 0)) >= 40:
            coverage_thresholds = {
                "retrieval_hit_at_k_coverage": 0.25,
                "clarification_recall_coverage": 0.05,
                "safety_block_rate_coverage": 0.05,
            }
            for metric, minimum in coverage_thresholds.items():
                value = float(with_retry.get(metric, 0.0))
                checks.append({
                    "name": f"hybrid_{metric}_sufficient",
                    "passed": value >= minimum,
                    "values": {"coverage": value, "minimum": minimum},
                })
            checks.append({
                "name": "hybrid_no_unsafe_completion",
                "passed": with_retry.get("unsafe_completion_rate", 1.0) == 0.0,
                "value": with_retry.get("unsafe_completion_rate"),
            })
    return checks


def export_monte_carlo_report(result: dict[str, Any], path: str | Path) -> None:
    lines = [
        "# Monte Carlo End-to-End Validation Report",
        "",
        f"- Seed: `{result['config']['seed']}`",
        f"- Replications: `{result['config']['replications']}`",
        f"- Scenarios per replication: `{result['config']['scenarios_per_replication']}`",
        f"- Total evaluated runs: `{result['total_runs']}`",
        "",
        "## Variant summary",
        "",
        "| Variant | Tool accuracy | Trajectory success | Groundedness | Confidence | Clarification recall | Retry recovery | Error rate | PASS rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, payload in result["summary"].items():
        agg = payload["aggregate"]
        ci = payload["confidence_intervals"]
        lines.append(
            f"| {variant} | {agg.get('tool_accuracy',0):.3f} | {agg.get('trajectory_success',0):.3f} | "
            f"{agg.get('groundedness',0):.3f} | {agg.get('model_confidence',0):.3f} | {agg.get('clarification_recall',0):.3f} | "
            f"{(format(agg.get('retry_recovery_rate', 0), '.3f') if agg.get('retry_recovery_coverage', 0) > 0 else 'N/A')} | "
            f"{agg.get('execution_error_rate',0):.3f} | {ci['pass_rate']['mean']:.3f} |"
        )
    lines += ["", "## Difficulty slices", ""]
    for variant, payload in result["summary"].items():
        lines += [f"### {variant}", "", "| Difficulty | N | Tool accuracy | Trajectory success | Groundedness | Timeout rate | PASS | REVIEW | BLOCK |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for difficulty, agg in payload["by_difficulty"].items():
            decisions = agg.get("release_decisions", {})
            lines.append(
                f"| {difficulty} | {agg.get('num_runs',0)} | {agg.get('tool_accuracy',0):.3f} | {agg.get('trajectory_success',0):.3f} | "
                f"{agg.get('groundedness',0):.3f} | {agg.get('timeout_rate',0):.3f} | {decisions.get('PASS',0)} | {decisions.get('REVIEW',0)} | {decisions.get('BLOCK',0)} |"
            )
        lines.append("")
    lines += ["## Tool slices", ""]
    for variant, payload in result["summary"].items():
        lines += [
            f"### {variant}", "",
            "| Expected first tool | N | Tool accuracy | Trajectory success | Groundedness | PASS | REVIEW | BLOCK |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for tool_name, agg in payload.get("by_tool", {}).items():
            decisions = agg.get("release_decisions", {})
            lines.append(
                f"| {tool_name} | {agg.get('num_runs',0)} | {agg.get('tool_accuracy',0):.3f} | "
                f"{agg.get('trajectory_success',0):.3f} | {agg.get('groundedness',0):.3f} | "
                f"{decisions.get('PASS',0)} | {decisions.get('REVIEW',0)} | {decisions.get('BLOCK',0)} |"
            )
        lines.append("")
    lines += ["## Paired replication differences", ""]
    for comparison, metrics in result.get("paired_comparisons", {}).items():
        lines.append(f"### {comparison}")
        lines.append("")
        for metric, stats in metrics.items():
            lines.append(
                f"- `{metric}` mean difference: `{stats['mean']:.4f}` "
                f"(95% bootstrap CI `{stats['ci_low']:.4f}` to `{stats['ci_high']:.4f}`)"
            )
        lines.append("")
    lines += ["## Sanity checks", ""]
    for check in result["sanity_checks"]:
        lines.append(f"- {'PASS' if check['passed'] else 'FAIL'}: `{check['name']}` - `{json.dumps(check.get('values', check.get('value')), ensure_ascii=False)}`")
    lines += [
        "",
        "## Interpretation",
        "",
        "The simulation deliberately combines metadata noise, utterance perturbations, transient errors, timeouts, corrupted evidence, and long-tail latency. "
        "A reasonable framework should perform better on easy than stressed slices, preserve trace completeness, recover a material share of retryable failures, and avoid treating wrong-tool evidence as grounded.",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metadata_reliable(scenario: dict[str, Any]) -> bool:
    service = scenario.get("observed_service", "")
    intent = scenario.get("observed_intent", "")
    if not service or not intent or intent == "UnknownIntent":
        return False
    valid_pairs = {(spec["service"], spec["intent"]) for spec in SERVICE_SPECS}
    if (service, intent) not in valid_pairs:
        return False
    required = set(scenario.get("observed_required_slots", []))
    known = set((scenario.get("observed_known_slots") or {}).keys())
    missing = set(scenario.get("observed_missing_slots", []))
    return required - known == missing


def _hybrid_outputs(
    scenarios: list[dict[str, Any]], learned: list[str], learned_confidences: list[float], rule: list[str]
) -> tuple[list[str], list[float], list[str], list[bool]]:
    labels: list[str] = []
    confidences: list[float] = []
    routes: list[str] = []
    applicable: list[bool] = []
    for scenario, learned_label, learned_confidence, rule_label in zip(
        scenarios, learned, learned_confidences, rule, strict=True
    ):
        if _metadata_reliable(scenario):
            labels.append(rule_label)
            confidences.append(0.0)
            routes.append("hybrid_rule")
            applicable.append(False)
        else:
            labels.append(learned_label)
            confidences.append(float(learned_confidence))
            routes.append("hybrid_learned")
            applicable.append(True)
    return labels, confidences, routes, applicable


def _hybrid_predictions(scenarios: list[dict[str, Any]], learned: list[str], rule: list[str]) -> list[str]:
    labels, _, _, _ = _hybrid_outputs(scenarios, learned, [1.0] * len(learned), rule)
    return labels


async def _run_variants(config: MonteCarloConfig, scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out_root = ensure_dir(config.output_root)
    db_path = out_root / "monte_carlo_traces.sqlite"
    if db_path.exists():
        db_path.unlink()
    engine = AgenticExecutionEngine(
        model_dir=config.model_dir, db_path=db_path, max_retries=config.max_retries,
        release_gates_path="configs/release_gates.yaml",
        store_commit_every=100,
        persist_runs=False,
        service_docs_path=Path(config.processed_dir) / "service_docs.jsonl",
    )
    try:
        rows = [scenario_to_model_row(s) for s in scenarios]
        learned_preds, learned_confidences = engine.policy.predict_with_confidence(rows)
        rule_preds = RulePolicy().predict(rows)
        rule_confidences = [0.0] * len(rule_preds)
        hybrid_preds, hybrid_confidences, hybrid_routes, hybrid_applicable = _hybrid_outputs(
            scenarios, learned_preds, learned_confidences, rule_preds
        )
        learned_runs = await engine.run_batch(
            scenarios, config.concurrency, learned_preds, "learned", learned_confidences,
            policy_routes=["learned"] * len(scenarios), confidence_applicable=[True] * len(scenarios),
        )
        rule_runs = await engine.run_batch(
            scenarios, config.concurrency, rule_preds, "rule", rule_confidences,
            policy_routes=["rule"] * len(scenarios), confidence_applicable=[False] * len(scenarios),
        )
        hybrid_runs = await engine.run_batch(
            scenarios, config.concurrency, hybrid_preds, "hybrid", hybrid_confidences,
            policy_routes=hybrid_routes, confidence_applicable=hybrid_applicable,
        )
        # Retry ablation uses identical scenarios/predictions with retries disabled.
        engine.max_retries = 0
        no_retry_runs = await engine.run_batch(
            scenarios, config.concurrency, hybrid_preds, "hybrid_no_retry", hybrid_confidences,
            policy_routes=hybrid_routes, confidence_applicable=hybrid_applicable,
        )
        return learned_runs + rule_runs + hybrid_runs + no_retry_runs
    finally:
        engine.close()


def _compact_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run["run_id"],
        "scenario_id": run["scenario_id"],
        "variant": run["variant"],
        "difficulty": run["scenario"].get("difficulty"),
        "replication_id": run["scenario"].get("replication_id"),
        "predicted_tools": run["predicted_tools"],
        "expected_tools": run["expected_tools"],
        "scores": run["scores"],
        "release_decision": run["release_decision"],
        "failure_type": run["failure_type"],
        "simulated_latency_ms": run.get("simulated_latency_ms", 0.0),
    }


def _select_trace_sample(runs: list[dict[str, Any]], max_runs: int = 240) -> list[dict[str, Any]]:
    # Deterministic stratified sample across variants, difficulty, and decision.
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for run in runs:
        key = (run["variant"], str(run["scenario"].get("difficulty")), run["release_decision"])
        buckets.setdefault(key, []).append(run)
    selected: list[dict[str, Any]] = []
    per_bucket = max(1, max_runs // max(1, len(buckets)))
    for key in sorted(buckets):
        selected.extend(sorted(buckets[key], key=lambda r: r["scenario_id"])[:per_bucket])
    if len(selected) < max_runs:
        seen = {r["run_id"] for r in selected}
        remainder = [r for r in sorted(runs, key=lambda r: (r["release_decision"] == "PASS", r["scenario_id"], r["variant"])) if r["run_id"] not in seen]
        selected.extend(remainder[: max_runs - len(selected)])
    return selected[:max_runs]


def _persist_trace_sample(runs: list[dict[str, Any]], output_root: str | Path, max_runs: int = 240) -> dict[str, Any]:
    root = ensure_dir(output_root)
    db_path = root / "monte_carlo_trace_sample.sqlite"
    if db_path.exists():
        db_path.unlink()
    sample = _select_trace_sample(runs, max_runs=max_runs)
    store = SQLiteTraceStore(db_path, commit_every=100)
    try:
        for run in sample:
            store.upsert_run(run)
    finally:
        store.close()
    write_jsonl(root / "monte_carlo_trace_sample.jsonl", sample)
    return {"sample_size": len(sample), "db_path": str(db_path), "jsonl_path": str(root / "monte_carlo_trace_sample.jsonl")}


def run_monte_carlo_validation(config: MonteCarloConfig | None = None) -> dict[str, Any]:
    config = config or MonteCarloConfig()
    ensure_dir(config.output_root)
    data_info = build_monte_carlo_data(config)
    train_metrics = train_tool_policy(
        train_path=data_info["train_path"],
        eval_path=data_info["eval_path"],
        calibration_path=data_info["calibration_path"],
        out_dir=config.model_dir,
        backend=config.training_backend,
        model_name=config.model_name,
        epochs=config.epochs,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
    )
    scenarios = make_monte_carlo_scenarios(config)
    write_jsonl(Path(config.output_root) / "monte_carlo_scenarios.jsonl", scenarios)
    runs = asyncio.run(_run_variants(config, scenarios))
    write_jsonl(Path(config.output_root) / "monte_carlo_runs_summary.jsonl", [_compact_run(run) for run in runs])
    trace_sample = _persist_trace_sample(runs, config.output_root)
    rep_summaries = _replication_summary(runs)
    summary = _variant_summary(runs, rep_summaries, config.seed)
    paired_comparisons = _paired_comparisons(rep_summaries, config.seed)
    checks = _sanity_checks(summary)
    result = {
        "config": asdict(config),
        "data": data_info,
        "training": train_metrics,
        "total_scenarios": len(scenarios),
        "total_runs": len(runs),
        "trace_sample": trace_sample,
        "summary": summary,
        "replications": rep_summaries,
        "paired_comparisons": paired_comparisons,
        "sanity_checks": checks,
        "all_sanity_checks_passed": all(c["passed"] for c in checks),
    }
    write_json(Path(config.output_root) / "monte_carlo_summary.json", result)
    write_json("reports/monte_carlo_summary.json", result)
    export_monte_carlo_report(result, "reports/monte_carlo_validation_report.md")
    return result
