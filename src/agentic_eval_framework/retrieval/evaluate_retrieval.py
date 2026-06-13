from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_eval_framework.retrieval.hierarchical_retriever import HierarchicalBM25Retriever
from agentic_eval_framework.retrieval.recency_query import current_utterance
from agentic_eval_framework.retrieval.targets import (
    build_retrieval_query,
    compatible_doc_ids,
    target_doc_id,
)
from agentic_eval_framework.utils.io import read_jsonl, write_json


def evaluate_retrieval(
    scenarios_path: str | Path = "data/processed/golden_trajectories.jsonl",
    docs_path: str | Path = "data/processed/service_docs.jsonl",
    out_path: str | Path = "outputs/retrieval_eval.json",
    top_k: int = 5,
    query_mode: str = "recency_weighted",
) -> dict[str, Any]:
    scenarios = read_jsonl(scenarios_path)
    docs = read_jsonl(docs_path)
    evidence_docs = [doc for doc in docs if doc.get("doc_type") in {"service", "intent"}]
    retriever = HierarchicalBM25Retriever(evidence_docs)
    available_ids = {str(doc.get("doc_id")) for doc in evidence_docs}
    strict_hits = compatible_hits = strict_top1 = compatible_top1 = 0
    strict_rr: list[float] = []
    compatible_rr: list[float] = []
    examples = []
    strategy = "recency_tool_filtered_bm25"
    retrieval_scenarios = [
        scenario
        for scenario in scenarios
        if (scenario.get("expected_tools") or [""])[0]
        in {"search_places", "calendar_lookup", "media_search", "weather_lookup", "search_docs"}
    ]

    for scenario in retrieval_scenarios:
        query = build_retrieval_query(scenario, mode=query_mode)
        tool_name = (scenario.get("expected_tools") or [None])[0]
        results = retriever.search(
            query,
            top_k=top_k,
            tool_name=tool_name,
            current_query=current_utterance(scenario),
        )
        if results:
            strategy = str(results[0].get("retrieval_strategy", strategy))
        target = target_doc_id(scenario, available_ids)
        compatible = compatible_doc_ids(scenario, evidence_docs)
        compatible.add(target)
        ids = [str(result["doc_id"]) for result in results]
        strict_rank = ids.index(target) + 1 if target in ids else None
        compatible_ranks = [index + 1 for index, doc_id in enumerate(ids) if doc_id in compatible]
        strict_hits += int(strict_rank is not None)
        compatible_hits += int(bool(compatible_ranks))
        strict_top1 += int(bool(ids) and ids[0] == target)
        compatible_top1 += int(bool(ids) and ids[0] in compatible)
        strict_rr.append(1.0 / strict_rank if strict_rank is not None else 0.0)
        compatible_rr.append(1.0 / min(compatible_ranks) if compatible_ranks else 0.0)
        examples.append(
            {
                "scenario_id": scenario["scenario_id"],
                "query": query,
                "target_doc_id": target,
                "compatible_doc_ids": sorted(compatible),
                "retrieved": ids,
            }
        )

    denominator = max(1, len(retrieval_scenarios))
    metrics = {
        "num_scenarios": len(scenarios),
        "num_retrieval_scenarios": len(retrieval_scenarios),
        "compatible_recall_at_k": compatible_hits / denominator,
        "strict_recall_at_k": strict_hits / denominator,
        "compatible_top1_accuracy": compatible_top1 / denominator,
        "strict_top1_accuracy": strict_top1 / denominator,
        "compatible_mrr": sum(compatible_rr) / max(1, len(compatible_rr)),
        "strict_mrr": sum(strict_rr) / max(1, len(strict_rr)),
        "top_k": top_k,
        "query_mode": query_mode,
        "strategy": strategy,
        "examples": examples[:50],
    }
    write_json(out_path, metrics)
    return metrics
