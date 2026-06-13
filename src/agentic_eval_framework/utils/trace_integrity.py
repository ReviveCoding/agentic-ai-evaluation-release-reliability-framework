from __future__ import annotations

from typing import Any

from agentic_eval_framework.utils.fingerprints import stable_json_hash


TRACE_HASH_FIELD = "trace_fingerprint"


def canonical_run_payload(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key != TRACE_HASH_FIELD}


def compute_trace_fingerprint(run: dict[str, Any]) -> str:
    return stable_json_hash(canonical_run_payload(run))


def attach_trace_fingerprint(run: dict[str, Any]) -> dict[str, Any]:
    payload = dict(run)
    payload[TRACE_HASH_FIELD] = compute_trace_fingerprint(payload)
    return payload


def verify_trace_fingerprint(run: dict[str, Any]) -> bool:
    stored = run.get(TRACE_HASH_FIELD)
    return bool(stored) and str(stored) == compute_trace_fingerprint(run)
