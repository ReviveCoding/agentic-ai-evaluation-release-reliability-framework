from __future__ import annotations

import asyncio
import hashlib
import math
import time
import uuid
from pathlib import Path
from typing import Any

from agentic_eval_framework.evaluators.ambiguity import clarification_recall, premature_action_rate
from agentic_eval_framework.evaluators.forbidden_tools import forbidden_tool_rate
from agentic_eval_framework.evaluators.groundedness import groundedness_score
from agentic_eval_framework.evaluators.safety import false_block_rate, safety_block_rate
from agentic_eval_framework.evaluators.retrieval_quality import retrieval_metrics
from agentic_eval_framework.evaluators.tool_accuracy import tool_accuracy
from agentic_eval_framework.evaluators.trajectory_success import exact_trajectory_success, partial_trajectory_success
from agentic_eval_framework.evaluators.terminal_safety import TERMINAL_TOOLS, completion_suppression_is_safe, unsafe_completion_rate
from agentic_eval_framework.engine.release_gate import decide_release, load_release_thresholds
from agentic_eval_framework.models.transformer_tool_policy import ToolPolicyModel
from agentic_eval_framework.retrieval.hierarchical_retriever import HierarchicalBM25Retriever
from agentic_eval_framework.retrieval.targets import build_retrieval_query, compatible_doc_ids, target_doc_id
from agentic_eval_framework.retrieval.recency_query import current_utterance
from agentic_eval_framework.storage.sqlite_store import SQLiteTraceStore
from agentic_eval_framework.tools.registry import TOOL_REGISTRY
from agentic_eval_framework.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl
from agentic_eval_framework.utils.trace_integrity import attach_trace_fingerprint

RETRIEVAL_TOOLS = {"search_places", "calendar_lookup", "media_search", "weather_lookup", "search_docs"}


class InjectedToolError(RuntimeError):
    def __init__(self, error_type: str, latency_ms: float) -> None:
        super().__init__(error_type)
        self.error_type = error_type
        self.latency_ms = latency_ms


def scenario_to_model_row(scenario: dict[str, Any]) -> dict[str, Any]:
    """Create the model input row while preserving hidden oracle labels separately.

    Monte Carlo scenarios may provide noisy observed metadata. Evaluators still
    use the true scenario fields, while the policy only sees observed fields.
    """
    return {
        "dialogue_context": scenario.get("dialogue_context", ""),
        "user_utterance": scenario.get("user_request", ""),
        "service": scenario.get("observed_service", scenario.get("service", "")),
        "intent": scenario.get("observed_intent", scenario.get("intent", "")),
        "service_description": scenario.get("observed_service_description", scenario.get("service_description", "")),
        "known_slots": scenario.get("observed_known_slots", scenario.get("known_slots", {})),
        "missing_slots": scenario.get("observed_missing_slots", scenario.get("missing_slots", [])),
        "required_slots": scenario.get("observed_required_slots", scenario.get("required_slots", [])),
        "risk_flags": scenario.get("observed_risk_flags", scenario.get("risk_flags", [])),
        "tool_label": scenario.get("expected_tools", [""])[0],
    }


def _stable_uniform(*parts: Any) -> float:
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
    return (value + 0.5) / (2**64)


def _fault_stream_key(profile: dict[str, Any], tool_name: str) -> str:
    """Return the random-stream key used for synthetic faults and latency.

    Monte Carlo policy variants are compared on matched scenarios. When
    ``couple_across_tools`` is enabled, every candidate *primary* tool sees the
    same latent fault and latency draw. This prevents a policy that selects a
    different tool from receiving an easier or harder random environment merely
    because the tool name changed. Terminal completion remains a separate stream.
    """
    if bool(profile.get("couple_across_tools", False)):
        return "terminal" if tool_name == "final_answer" else "primary"
    return tool_name


def _synthetic_latency_ms(profile: dict[str, Any], scenario_id: str, tool_name: str, attempt: int) -> float:
    base = float(profile.get("latency_base_ms", 8.0))
    jitter = float(profile.get("latency_jitter_ms", 12.0))
    stream_key = _fault_stream_key(profile, tool_name)
    u = min(
        max(_stable_uniform(profile.get("seed", 0), scenario_id, stream_key, attempt, "latency"), 1e-12),
        1 - 1e-12,
    )
    return max(0.01, base + jitter * (-math.log(1.0 - u)))


class AgenticExecutionEngine:
    def __init__(
        self,
        model_dir: str | Path = "models/tool_policy",
        db_path: str | Path = "outputs/traces.sqlite",
        timeout_s: float = 2.0,
        max_retries: int = 2,
        retry_backoff_s: float = 0.0,
        release_gates_path: str | Path = "configs/release_gates.yaml",
        store_commit_every: int = 1,
        persist_runs: bool = True,
        service_docs_path: str | Path = "data/processed/service_docs.jsonl",
        retrieval_top_k: int = 5,
        retrieval_query_mode: str = "recency_weighted",
    ) -> None:
        self.policy = ToolPolicyModel(model_dir)
        self.store = SQLiteTraceStore(db_path, commit_every=store_commit_every)
        self.timeout_s = timeout_s
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self.thresholds = load_release_thresholds(release_gates_path)
        if self.policy.recommended_min_confidence is not None:
            self.thresholds["min_model_confidence"] = max(
                self.thresholds["min_model_confidence"], self.policy.recommended_min_confidence
            )
        self.persist_runs = bool(persist_runs)
        self.service_docs_path = Path(service_docs_path)
        self.retrieval_top_k = max(1, int(retrieval_top_k))
        if retrieval_query_mode not in {"recency_weighted", "utterance_context", "metadata_enriched"}:
            raise ValueError("retrieval_query_mode must be recency_weighted, utterance_context, or metadata_enriched")
        self.retrieval_query_mode = retrieval_query_mode
        docs = read_jsonl(self.service_docs_path) if self.service_docs_path.exists() else []
        evidence_docs = [doc for doc in docs if doc.get("doc_type") in {"service", "intent"}]
        self.retrieval_docs = evidence_docs
        self.retrieval_doc_ids = {str(doc.get("doc_id")) for doc in evidence_docs}
        self.retriever = HierarchicalBM25Retriever(evidence_docs) if evidence_docs else None

    def close(self) -> None:
        self.store.close()

    def _fault_type(self, scenario: dict[str, Any], tool_name: str, attempt: int) -> str | None:
        profile = scenario.get("fault_profile", {}) or {}
        if not profile:
            return None
        seed = profile.get("seed", scenario.get("simulation_seed", 0))
        sid = scenario.get("scenario_id", "unknown")
        stream_key = _fault_stream_key(profile, tool_name)
        u = _stable_uniform(seed, sid, stream_key, attempt, "fault")
        p_timeout = float(profile.get("timeout_prob", 0.0))
        p_transient = float(profile.get("transient_error_prob", 0.0))
        p_corrupt = float(profile.get("corrupt_evidence_prob", 0.0))
        if u < p_timeout:
            return "timeout"
        if u < p_timeout + p_transient:
            return "transient_error"
        if u < p_timeout + p_transient + p_corrupt:
            return "corrupt_evidence"
        return None

    @staticmethod
    def _tool_kwargs(tool_name: str, scenario: dict[str, Any], previous_obs: dict[str, Any] | None) -> dict[str, Any]:
        observed_service = scenario.get("observed_service", scenario.get("service", ""))
        kwargs: dict[str, Any] = {"query": scenario.get("user_request", ""), "service": observed_service}
        if tool_name == "ask_clarification":
            kwargs = {"query": scenario.get("user_request", ""), "missing_slots": scenario.get("missing_slots", [])}
        elif tool_name == "safety_check":
            kwargs = {"query": scenario.get("user_request", ""), "risk_flags": scenario.get("risk_flags", [])}
        elif tool_name == "final_answer":
            kwargs = {"query": scenario.get("user_request", ""), "evidence": previous_obs or {}}
        return kwargs

    @staticmethod
    def _validate_observation(tool_name: str, scenario: dict[str, Any], observation: dict[str, Any], previous_obs: dict[str, Any] | None) -> dict[str, Any]:
        obs = dict(observation)
        expected_first = (scenario.get("expected_tools") or [None])[0]
        if tool_name == "final_answer":
            compatible = bool((previous_obs or {}).get("evidence_compatible", False))
        else:
            compatible = tool_name == expected_first
            if "retrieval_target_doc_id" in obs:
                compatible_ids = set(obs.get("retrieval_compatible_doc_ids") or [obs.get("retrieval_target_doc_id")])
                compatible = compatible and obs.get("evidence_id") in compatible_ids
        obs["evidence_compatible"] = compatible
        if not compatible:
            obs["grounded"] = False
            obs["compatibility_error"] = (
                f"tool={tool_name} expected={expected_first} "
                f"evidence={obs.get('evidence_id')} target={obs.get('retrieval_target_doc_id')}"
            )
        return obs

    def _attach_retrieval_evidence(
        self, tool_name: str, scenario: dict[str, Any], observation: dict[str, Any]
    ) -> dict[str, Any]:
        if tool_name not in RETRIEVAL_TOOLS or self.retriever is None:
            return observation
        query = build_retrieval_query(scenario, mode=self.retrieval_query_mode)
        results = self.retriever.search(
            query,
            top_k=self.retrieval_top_k,
            tool_name=tool_name,
            current_query=current_utterance(scenario),
        )
        ids = [str(item.get("doc_id")) for item in results]
        target = target_doc_id(scenario, self.retrieval_doc_ids)
        obs = dict(observation)
        compatible_ids = compatible_doc_ids(scenario, self.retrieval_docs)
        if target in self.retrieval_doc_ids:
            compatible_ids.add(target)
        obs.update({
            "retrieval_query": query,
            "retrieval_query_mode": self.retrieval_query_mode,
            "retrieved_doc_ids": ids,
            "retrieval_scores": [float(item.get("score", 0.0)) for item in results],
            "retrieval_target_doc_id": target,
            "retrieval_compatible_doc_ids": sorted(compatible_ids),
            "evidence_id": ids[0] if ids else "none",
            "retrieval_hit_at_k": any(doc_id in compatible_ids for doc_id in ids),
            "retrieval_strict_hit_at_k": target in ids,
            "retrieval_strategy": results[0].get("retrieval_strategy", "hierarchical_bm25_rrf") if results else "hierarchical_bm25_rrf",
            "retrieval_service_shortlist": results[0].get("service_shortlist", []) if results else [],
        })
        if results:
            obs["summary"] = str(results[0].get("text", obs.get("summary", "")))
        return obs

    async def _execute_tool_once(
        self,
        tool_name: str,
        scenario: dict[str, Any],
        previous_obs: dict[str, Any] | None,
        attempt: int,
    ) -> tuple[dict[str, Any], float, str | None]:
        profile = scenario.get("fault_profile", {}) or {}
        synthetic_latency = _synthetic_latency_ms(profile, scenario.get("scenario_id", "unknown"), tool_name, attempt)
        fault = self._fault_type(scenario, tool_name, attempt)
        if fault in {"timeout", "transient_error"}:
            raise InjectedToolError(fault, synthetic_latency)

        start = time.perf_counter()
        if tool_name not in TOOL_REGISTRY:
            observation = {
                "tool_name": tool_name,
                "evidence_type": "execution_error",
                "evidence_id": "error",
                "summary": f"Unknown tool requested: {tool_name}",
                "grounded": False,
                "evidence_compatible": False,
                "unknown_tool": True,
            }
            return observation, synthetic_latency if profile else 0.0, "unknown_tool"
        tool_fn = TOOL_REGISTRY[tool_name]
        kwargs = self._tool_kwargs(tool_name, scenario, previous_obs)
        await asyncio.sleep(0)
        if profile:
            # Monte Carlo tools are deterministic, CPU-light local functions.
            # Calling them directly avoids creating thousands of unnecessary
            # thread-pool jobs while still exercising the same tool interface.
            observation = tool_fn(**kwargs)
        else:
            observation = await asyncio.wait_for(asyncio.to_thread(tool_fn, **kwargs), timeout=self.timeout_s)
        actual_ms = (time.perf_counter() - start) * 1000
        latency_ms = synthetic_latency if profile else actual_ms
        observation = self._attach_retrieval_evidence(tool_name, scenario, observation)
        observation = self._validate_observation(tool_name, scenario, observation, previous_obs)
        if fault == "corrupt_evidence":
            observation.update({
                "evidence_id": "corrupt",
                "grounded": False,
                "evidence_compatible": False,
                "corruption_injected": True,
            })
        return observation, latency_ms, fault

    async def _execute_tool(
        self,
        tool_name: str,
        scenario: dict[str, Any],
        previous_obs: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], float, dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        total_latency = 0.0
        for attempt in range(self.max_retries + 1):
            try:
                obs, latency_ms, fault = await self._execute_tool_once(tool_name, scenario, previous_obs, attempt)
                total_latency += latency_ms
                attempts.append({"attempt": attempt + 1, "status": "success", "fault": fault, "latency_ms": latency_ms})
                final_error = bool(obs.get("unknown_tool"))
                return obs, total_latency, {
                    "attempts": attempts,
                    "retry_count": attempt,
                    "recovered_after_retry": attempt > 0,
                    "timeout_count": sum(a.get("fault") == "timeout" for a in attempts),
                    "transient_error_count": sum(a.get("fault") == "transient_error" for a in attempts),
                    "final_execution_error": final_error,
                }
            except InjectedToolError as exc:
                total_latency += exc.latency_ms
                attempts.append({"attempt": attempt + 1, "status": "error", "fault": exc.error_type, "latency_ms": exc.latency_ms})
            except asyncio.TimeoutError:
                total_latency += self.timeout_s * 1000
                attempts.append({"attempt": attempt + 1, "status": "error", "fault": "timeout", "latency_ms": self.timeout_s * 1000})
            except Exception as exc:
                attempts.append({"attempt": attempt + 1, "status": "error", "fault": "execution_error", "message": str(exc), "latency_ms": 0.0})
            if attempt < self.max_retries and self.retry_backoff_s:
                await asyncio.sleep(self.retry_backoff_s * (2**attempt))

        last_fault = attempts[-1].get("fault", "execution_error") if attempts else "execution_error"
        obs = {
            "tool_name": tool_name,
            "evidence_type": "error",
            "evidence_id": "error",
            "summary": f"Tool execution failed after {len(attempts)} attempts: {last_fault}",
            "grounded": False,
            "evidence_compatible": False,
            "execution_error": last_fault,
        }
        return obs, total_latency, {
            "attempts": attempts,
            "retry_count": max(0, len(attempts) - 1),
            "recovered_after_retry": False,
            "timeout_count": sum(a.get("fault") == "timeout" for a in attempts),
            "transient_error_count": sum(a.get("fault") == "transient_error" for a in attempts),
            "final_execution_error": True,
        }

    async def run_scenario(
        self,
        scenario: dict[str, Any],
        predicted_first: str | None = None,
        variant: str = "learned",
        policy_confidence: float | None = None,
        policy_route: str | None = None,
        confidence_applicable: bool | None = None,
    ) -> dict[str, Any]:
        run_id = f"{variant}_{uuid.uuid4().hex[:12]}"
        start = time.perf_counter()
        row = scenario_to_model_row(scenario)
        if predicted_first is None:
            labels, confidences = self.policy.predict_with_confidence([row])
            predicted_first = labels[0]
            policy_confidence = confidences[0]
            policy_route = policy_route or "learned"
            confidence_applicable = True if confidence_applicable is None else confidence_applicable
        else:
            policy_route = policy_route or variant
            if confidence_applicable is None:
                confidence_applicable = policy_route in {"learned", "api", "hybrid_learned"}
            if policy_confidence is None:
                policy_confidence = 0.0 if not confidence_applicable else 1.0

        expected_tools = scenario.get("expected_tools", [])
        predicted_tools: list[str] = []
        steps: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []

        async def execute_step(tool_name: str, previous_obs: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
            step_id = f"{run_id}_step_{len(steps) + 1}"
            obs, latency_ms, execution_meta = await self._execute_tool(tool_name, scenario, previous_obs=previous_obs)
            predicted_tools.append(tool_name)
            observations.append(obs)
            steps.append({
                "step_id": step_id,
                "predicted_tool": tool_name,
                "latency_ms": latency_ms,
                "observation": obs,
                "execution": execution_meta,
            })
            return obs, execution_meta

        first_obs, first_execution = await execute_step(predicted_first, None)
        completion_suppressed = False
        completion_reason: str | None = None
        if predicted_first not in TERMINAL_TOOLS:
            evidence_valid = (
                bool(first_obs.get("grounded"))
                and bool(first_obs.get("evidence_compatible"))
                and not bool(first_execution.get("final_execution_error"))
            )
            if evidence_valid:
                await execute_step("final_answer", first_obs)
            else:
                completion_suppressed = True
                completion_reason = (
                    "execution_error" if first_execution.get("final_execution_error")
                    else "invalid_or_incompatible_evidence"
                )

        latency_ms = (time.perf_counter() - start) * 1000
        total_attempts = sum(len(step["execution"].get("attempts", [])) for step in steps)
        total_timeouts = sum(step["execution"].get("timeout_count", 0) for step in steps)
        retried_steps = sum(step["execution"].get("retry_count", 0) > 0 for step in steps)
        recovered_steps = sum(bool(step["execution"].get("recovered_after_retry")) for step in steps)
        failed_steps = sum(bool(step["execution"].get("final_execution_error")) for step in steps)
        retrieval = retrieval_metrics(observations)
        confidence_value = float(policy_confidence or 0.0)
        scores = {
            "tool_accuracy": tool_accuracy(predicted_tools, expected_tools),
            "trajectory_success": exact_trajectory_success(predicted_tools, expected_tools),
            "partial_trajectory_success": partial_trajectory_success(predicted_tools, expected_tools),
            "groundedness": groundedness_score(observations),
            "clarification_applicable": 1.0 if scenario.get("missing_slots", []) else 0.0,
            "clarification_recall": clarification_recall(predicted_tools, scenario.get("missing_slots", [])),
            "premature_action_rate": premature_action_rate(predicted_tools, scenario.get("missing_slots", [])),
            "safety_applicable": 1.0 if scenario.get("risk_flags", []) else 0.0,
            "safety_block_rate": safety_block_rate(predicted_tools, scenario.get("risk_flags", [])),
            "false_block_applicable": 0.0 if scenario.get("risk_flags", []) else 1.0,
            "false_block_rate": false_block_rate(predicted_tools, scenario.get("risk_flags", [])),
            "forbidden_tool_rate": forbidden_tool_rate(predicted_tools, scenario.get("must_not_tools", [])),
            "review_required": 1.0 if scenario.get("release_gate_expected") == "REVIEW" else 0.0,
            "model_confidence": confidence_value,
            "confidence_applicable": 1.0 if confidence_applicable else 0.0,
            "trace_completeness": 1.0 if all(step.get("observation") and step.get("execution") for step in steps) else 0.0,
            "evidence_compatibility": sum(bool(obs.get("evidence_compatible")) for obs in observations) / max(1, len(observations)),
            "execution_error_rate": failed_steps / max(1, len(steps)),
            "timeout_rate": total_timeouts / max(1, total_attempts),
            "retry_rate": retried_steps / max(1, len(steps)),
            "retry_recovery_applicable": 1.0 if retried_steps else 0.0,
            "retry_recovery_rate": recovered_steps / retried_steps if retried_steps else 0.0,
            "max_step_latency_ms": max((step["latency_ms"] for step in steps), default=latency_ms),
            "unsafe_completion_rate": unsafe_completion_rate(steps),
            "safe_completion_behavior": completion_suppression_is_safe(
                predicted_first, first_obs, first_execution, predicted_tools
            ),
            **retrieval,
        }
        decision, failure_type = decide_release(scores, self.thresholds)
        status = "SUCCEEDED" if decision == "PASS" else ("BLOCKED" if decision == "BLOCK" else "REVIEW")
        run = {
            "run_id": run_id,
            "scenario_id": scenario["scenario_id"],
            "variant": variant,
            "policy_route": policy_route,
            "confidence_applicable": bool(confidence_applicable),
            "status": status,
            "model_backend": self.policy.backend if policy_route in {"learned", "api", "hybrid_learned"} else policy_route,
            "model_fingerprint": self.policy.model_fingerprint,
            "model_confidence_threshold": self.thresholds["min_model_confidence"],
            "predicted_tools": predicted_tools,
            "expected_tools": expected_tools,
            "completion_suppressed": completion_suppressed,
            "completion_reason": completion_reason,
            "steps": steps,
            "scores": scores,
            "release_decision": decision,
            "failure_type": failure_type,
            "latency_ms": latency_ms,
            "simulated_latency_ms": sum(step["latency_ms"] for step in steps),
            "execution_stats": {
                "total_steps": len(steps),
                "total_attempts": total_attempts,
                "total_timeouts": total_timeouts,
                "retried_steps": retried_steps,
                "recovered_steps": recovered_steps,
                "failed_steps": failed_steps,
            },
            "scenario": scenario,
        }
        run = attach_trace_fingerprint(run)
        if self.persist_runs:
            self.store.upsert_run(run)
        return run

    async def run_batch(
        self,
        scenarios: list[dict[str, Any]],
        concurrency: int = 8,
        predicted_first_tools: list[str] | None = None,
        variant: str = "learned",
        predicted_confidences: list[float] | None = None,
        policy_routes: list[str] | None = None,
        confidence_applicable: list[bool] | None = None,
    ) -> list[dict[str, Any]]:
        sem = asyncio.Semaphore(concurrency)
        if predicted_first_tools is None:
            model_rows = [scenario_to_model_row(scenario) for scenario in scenarios]
            predicted_first_tools, predicted_confidences = self.policy.predict_with_confidence(model_rows)
        if len(predicted_first_tools) != len(scenarios):
            raise ValueError("predicted_first_tools length must match scenarios length")
        if predicted_confidences is None:
            predicted_confidences = [1.0] * len(scenarios)
        if len(predicted_confidences) != len(scenarios):
            raise ValueError("predicted_confidences length must match scenarios length")
        if policy_routes is None:
            policy_routes = [variant] * len(scenarios)
        if len(policy_routes) != len(scenarios):
            raise ValueError("policy_routes length must match scenarios length")
        if confidence_applicable is None:
            confidence_applicable = [route in {"learned", "api", "hybrid_learned"} for route in policy_routes]
        if len(confidence_applicable) != len(scenarios):
            raise ValueError("confidence_applicable length must match scenarios length")

        async def _run(
            scenario: dict[str, Any], predicted_first: str, confidence: float, route: str, applicable: bool
        ) -> dict[str, Any]:
            async with sem:
                return await self.run_scenario(
                    scenario, predicted_first=predicted_first, variant=variant,
                    policy_confidence=confidence, policy_route=route, confidence_applicable=applicable,
                )

        return await asyncio.gather(*[
            _run(scenario, predicted, confidence, route, applicable)
            for scenario, predicted, confidence, route, applicable in zip(
                scenarios, predicted_first_tools, predicted_confidences, policy_routes, confidence_applicable, strict=True
            )
        ])


def run_evaluation(
    scenarios_path: str | Path = "data/processed/golden_trajectories.jsonl",
    model_dir: str | Path = "models/tool_policy",
    out_path: str | Path = "outputs/evaluation_results.json",
    traces_path: str | Path = "outputs/traces.jsonl",
    db_path: str | Path = "outputs/traces.sqlite",
    concurrency: int = 8,
    append: bool = False,
    release_gates_path: str | Path = "configs/release_gates.yaml",
    service_docs_path: str | Path = "data/processed/service_docs.jsonl",
) -> dict[str, Any]:
    scenarios = read_jsonl(scenarios_path)
    if not append:
        for artifact in [Path(db_path), Path(traces_path), Path(out_path)]:
            if artifact.exists():
                artifact.unlink()
    engine = AgenticExecutionEngine(
        model_dir=model_dir, db_path=db_path, release_gates_path=release_gates_path,
        service_docs_path=service_docs_path,
    )
    try:
        runs = asyncio.run(engine.run_batch(scenarios, concurrency=concurrency))
    finally:
        engine.close()
    ensure_dir(Path(out_path).parent)
    write_jsonl(traces_path, runs)
    aggregate = aggregate_runs(runs)
    result = {"aggregate": aggregate, "runs": runs}
    write_json(out_path, result)
    return result


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    if lo == hi:
        return float(values[lo])
    frac = pos - lo
    return float(values[lo] * (1 - frac) + values[hi] * frac)


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    weighted_metrics = {"execution_error_rate", "timeout_rate", "retry_rate", "retry_recovery_rate"}
    conditional_metrics = {
        "model_confidence": "confidence_applicable",
        "retrieval_hit_at_k": "retrieval_applicable",
        "retrieval_strict_hit_at_k": "retrieval_applicable",
        "retrieval_top1_accuracy": "retrieval_applicable",
        "retrieval_strict_top1_accuracy": "retrieval_applicable",
        "retrieval_reciprocal_rank": "retrieval_applicable",
        "retrieval_strict_reciprocal_rank": "retrieval_applicable",
        "clarification_recall": "clarification_applicable",
        "premature_action_rate": "clarification_applicable",
        "safety_block_rate": "safety_applicable",
        "false_block_rate": "false_block_applicable",
    }
    score_names = sorted({k for run in runs for k in run.get("scores", {}).keys() if k not in {"max_step_latency_ms", "p95_latency_ms"}})
    agg = {
        name: sum(run["scores"].get(name, 0.0) for run in runs) / len(runs)
        for name in score_names if name not in weighted_metrics | set(conditional_metrics)
    }
    for metric, applicability in conditional_metrics.items():
        applicable_runs = [
            run for run in runs if run.get("scores", {}).get(applicability, 0.0) >= 0.5
        ]
        agg[metric] = (
            sum(run["scores"].get(metric, 0.0) for run in applicable_runs) / len(applicable_runs)
            if applicable_runs else 0.0
        )
        coverage_name = "confidence_coverage" if metric == "model_confidence" else f"{metric}_coverage"
        agg[coverage_name] = len(applicable_runs) / len(runs)
    stats = [run.get("execution_stats", {}) for run in runs]
    total_steps = sum(int(s.get("total_steps", 0)) for s in stats)
    total_attempts = sum(int(s.get("total_attempts", 0)) for s in stats)
    total_timeouts = sum(int(s.get("total_timeouts", 0)) for s in stats)
    retried_steps = sum(int(s.get("retried_steps", 0)) for s in stats)
    recovered_steps = sum(int(s.get("recovered_steps", 0)) for s in stats)
    failed_steps = sum(int(s.get("failed_steps", 0)) for s in stats)
    agg["execution_error_rate"] = failed_steps / max(1, total_steps)
    agg["timeout_rate"] = total_timeouts / max(1, total_attempts)
    agg["retry_rate"] = retried_steps / max(1, total_steps)
    agg["retry_recovery_rate"] = recovered_steps / retried_steps if retried_steps else 0.0
    agg["retry_recovery_coverage"] = retried_steps / max(1, total_steps)
    step_latencies = [float(step.get("latency_ms", 0.0)) for run in runs for step in run.get("steps", [])]
    run_latencies = [float(run.get("simulated_latency_ms", run.get("latency_ms", 0.0))) for run in runs]
    agg["p50_step_latency_ms"] = percentile(step_latencies, 0.50)
    agg["p95_step_latency_ms"] = percentile(step_latencies, 0.95)
    agg["p99_step_latency_ms"] = percentile(step_latencies, 0.99)
    agg["p95_run_latency_ms"] = percentile(run_latencies, 0.95)
    decisions: dict[str, int] = {}
    failure_types: dict[str, int] = {}
    for run in runs:
        decisions[run["release_decision"]] = decisions.get(run["release_decision"], 0) + 1
        failure = str(run.get("failure_type") or "none")
        failure_types[failure] = failure_types.get(failure, 0) + 1
    agg["num_runs"] = len(runs)
    agg["release_decisions"] = decisions
    agg["failure_types"] = failure_types
    return agg
