from __future__ import annotations

import asyncio
from collections import Counter
import math
from pathlib import Path
import random
from typing import Any

import numpy as np

from agentic_eval_framework.engine.execution_engine import AgenticExecutionEngine
from agentic_eval_framework.storage.lineage import reconstruct_lineage
from agentic_eval_framework.storage.sqlite_store import SQLiteTraceStore
from agentic_eval_framework.utils.io import write_json
from agentic_eval_framework.utils.trace_integrity import verify_trace_fingerprint

NONDETERMINISTIC_SCORE_KEYS = {
    "max_step_latency_ms",
    "p50_step_latency_ms",
    "p95_step_latency_ms",
    "p99_step_latency_ms",
    "p95_run_latency_ms",
    "p95_latency_ms",
}
FLOAT_ABS_TOL = 1e-5
FLOAT_REL_TOL = 1e-5


def _set_replay_determinism(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _stable_scores(scores: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in scores.items() if key not in NONDETERMINISTIC_SCORE_KEYS}


def _step_signature(run: dict[str, Any]) -> list[dict[str, Any]]:
    signature: list[dict[str, Any]] = []
    for step in run.get("steps", []) or []:
        observation = step.get("observation", {}) or {}
        execution = step.get("execution", {}) or {}
        signature.append(
            {
                "tool_name": step.get("tool_name"),
                "execution_status": execution.get("status"),
                "final_error": execution.get("final_error"),
                "retry_count": execution.get("retry_count"),
                "recovered_after_retry": execution.get("recovered_after_retry"),
                "evidence_id": observation.get("evidence_id"),
                "retrieved_doc_ids": list(observation.get("retrieved_doc_ids") or []),
                "retrieval_target_doc_id": observation.get("retrieval_target_doc_id"),
                "evidence_compatible": observation.get("evidence_compatible"),
                "grounded": observation.get("grounded"),
            }
        )
    return signature


def _equivalent(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL)
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(_equivalent(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_equivalent(a, b) for a, b in zip(left, right))
    return left == right


def _compare_replay(original: dict[str, Any], replayed: dict[str, Any]) -> dict[str, Any]:
    replay_mode = str(replayed.get("replay_mode", "execution_only"))
    original_fp = original.get("model_fingerprint")
    replay_fp = replayed.get("model_fingerprint")
    checks: dict[str, bool] = {
        "predicted_tools_match": original.get("predicted_tools") == replayed.get("predicted_tools"),
        "release_decision_match": original.get("release_decision") == replayed.get("release_decision"),
        "failure_type_match": original.get("failure_type") == replayed.get("failure_type"),
        "completion_suppressed_match": original.get("completion_suppressed") == replayed.get("completion_suppressed"),
        "stable_step_signature_match": _equivalent(_step_signature(original), _step_signature(replayed)),
        "stable_scores_match": _equivalent(
            _stable_scores(original.get("scores", {})),
            _stable_scores(replayed.get("scores", {})),
        ),
    }
    if replay_mode == "full_policy_reexecution":
        checks["model_fingerprint_match"] = bool(
            original_fp and replay_fp and original_fp == replay_fp
        )
    return {"verified": all(checks.values()), "checks": checks}


async def _reexecute_runs(
    failed: list[dict[str, Any]],
    model_dir: str | Path,
    temp_db_path: str | Path,
    release_gates_path: str | Path,
    max_retries: int,
    service_docs_path: str | Path,
) -> list[dict[str, Any]]:
    _set_replay_determinism()
    engine = AgenticExecutionEngine(
        model_dir=model_dir,
        db_path=temp_db_path,
        release_gates_path=release_gates_path,
        max_retries=max_retries,
        persist_runs=False,
        service_docs_path=service_docs_path,
    )
    replayed: list[dict[str, Any]] = []
    try:
        for original in failed:
            scenario = original.get("scenario") or {}
            variant = str(original.get("variant", "learned"))
            can_repredict = (
                variant in {"learned", "api"}
                and original.get("model_fingerprint") == engine.policy.model_fingerprint
            )
            predicted_tools = original.get("predicted_tools") or []
            predicted_first = None if can_repredict else (predicted_tools[0] if predicted_tools else None)
            rerun = await engine.run_scenario(
                scenario,
                predicted_first=predicted_first,
                variant=variant,
            )
            rerun["replay_mode"] = "full_policy_reexecution" if can_repredict else "execution_only"
            replayed.append(rerun)
    finally:
        engine.close()
    return replayed


def replay_failed_runs(
    db_path: str | Path = "outputs/traces.sqlite",
    out_path: str | Path = "outputs/replay_failures.json",
    *,
    reexecute: bool = True,
    model_dir: str | Path = "models/tool_policy",
    release_gates_path: str | Path = "configs/release_gates.yaml",
    max_retries: int = 2,
    service_docs_path: str | Path = "data/processed/service_docs.jsonl",
) -> dict[str, Any]:
    store = SQLiteTraceStore(db_path)
    try:
        failed = store.list_failed_runs()
    finally:
        store.close()

    reruns: list[dict[str, Any]] = []
    temp_db = Path(out_path).with_name("replay_temp.sqlite")
    if reexecute and failed:
        if temp_db.exists():
            temp_db.unlink()
        reruns = asyncio.run(
            _reexecute_runs(
                failed, model_dir, temp_db, release_gates_path, max_retries, service_docs_path
            )
        )
        if temp_db.exists():
            temp_db.unlink()

    replay: list[dict[str, Any]] = []
    mismatch_counts: Counter[str] = Counter()
    mode_totals: Counter[str] = Counter()
    mode_verified: Counter[str] = Counter()

    for idx, run in enumerate(failed):
        item: dict[str, Any] = {
            "run_id": run["run_id"],
            "original_trace_integrity_valid": verify_trace_fingerprint(run),
            "scenario_id": run["scenario_id"],
            "release_decision": run.get("release_decision"),
            "failure_type": run.get("failure_type"),
            "predicted_tools": run.get("predicted_tools"),
            "expected_tools": run.get("expected_tools"),
            "model_fingerprint": run.get("model_fingerprint"),
            "scores": run.get("scores"),
            "lineage": reconstruct_lineage(run),
        }
        if reexecute:
            rerun = reruns[idx]
            comparison = _compare_replay(run, rerun)
            trace_valid = verify_trace_fingerprint(run)
            mode = str(rerun.get("replay_mode", "execution_only"))
            verified = bool(trace_valid and comparison["verified"])
            mode_totals[mode] += 1
            mode_verified[mode] += int(verified)
            for check, passed in comparison["checks"].items():
                if not passed:
                    mismatch_counts[check] += 1
            if not trace_valid:
                mismatch_counts["original_trace_integrity_invalid"] += 1
            item.update(
                {
                    "replay_verified": verified,
                    "replay_mode": mode,
                    "replay_checks": comparison["checks"],
                    "replayed_model_fingerprint": rerun.get("model_fingerprint"),
                    "replayed_release_decision": rerun.get("release_decision"),
                    "replayed_failure_type": rerun.get("failure_type"),
                    "replayed_predicted_tools": rerun.get("predicted_tools"),
                }
            )
        replay.append(item)

    verified_count = sum(bool(item.get("replay_verified")) for item in replay) if reexecute else 0
    mode_summary = {
        mode: {
            "count": mode_totals[mode],
            "verified": mode_verified[mode],
            "verification_rate": mode_verified[mode] / mode_totals[mode] if mode_totals[mode] else None,
        }
        for mode in sorted(mode_totals)
    }
    result = {
        "num_failed_or_review_runs": len(replay),
        "reexecution_enabled": reexecute,
        "num_replay_verified": verified_count,
        "replay_verification_rate": verified_count / len(replay) if replay and reexecute else (1.0 if reexecute else None),
        "mode_summary": mode_summary,
        "mismatch_counts": dict(mismatch_counts.most_common()),
        "float_abs_tolerance": FLOAT_ABS_TOL,
        "float_rel_tolerance": FLOAT_REL_TOL,
        "replay": replay,
    }
    write_json(out_path, result)
    return result
