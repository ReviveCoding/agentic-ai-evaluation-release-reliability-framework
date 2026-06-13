"""Optional LLM judge hook, disabled by default.

Default framework scoring uses deterministic evaluators for reproducibility. This
interface is intentionally minimal so a user can add an external local or hosted
LLM judge later without changing the release-gate pipeline.
"""

from __future__ import annotations


def judge_answer(prompt: str, answer: str, evidence: str) -> dict:
    raise NotImplementedError("Optional LLM judge is disabled by default.")
