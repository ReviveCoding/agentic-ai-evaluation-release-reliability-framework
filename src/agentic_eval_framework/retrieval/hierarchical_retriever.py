from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_eval_framework.retrieval.bm25_retriever import BM25Retriever
from agentic_eval_framework.retrieval.top1_reranker import (
    LightweightIntentReranker,
    build_feature_vector,
    doc_tool_label,
    infer_query_tool,
    service_base,
)


def _rrf(rank: int | None, k: int = 60) -> float:
    return 0.0 if rank is None else 1.0 / (k + rank)


@dataclass
class HierarchicalBM25Retriever:
    """Recency-aware two-stage BM25 retrieval with optional lightweight reranking.

    The primary candidate pool uses predicted-tool compatibility and likely
    services. Global lexical candidates remain as a fallback, preventing a wrong
    policy prediction from making the correct evidence unreachable.
    """

    docs: list[dict[str, Any]]
    service_top_k: int = 8
    candidate_top_k: int = 40
    reranker_path: str | Path = "models/retrieval_reranker.joblib"

    def __post_init__(self) -> None:
        self.docs = [dict(doc) for doc in self.docs]
        self.service_docs = [doc for doc in self.docs if doc.get("doc_type") == "service"]
        self.global_retriever = BM25Retriever(self.docs)
        self.service_retriever = BM25Retriever(self.service_docs) if self.service_docs else None
        self.docs_by_service: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for doc in self.docs:
            self.docs_by_service[str(doc.get("service", ""))].append(doc)
        self.reranker = LightweightIntentReranker.load(self.reranker_path)

    def _candidate_rows(
        self,
        query: str,
        *,
        current_query: str,
        tool_name: str | None,
    ) -> list[dict[str, Any]]:
        recency_results = self.global_retriever.search(query, top_k=min(len(self.docs), self.candidate_top_k))
        current_results = self.global_retriever.search(
            current_query or query,
            top_k=min(len(self.docs), self.candidate_top_k),
        )
        recency_rank = {str(item.get("doc_id")): int(item["rank"]) for item in recency_results}
        current_rank = {str(item.get("doc_id")): int(item["rank"]) for item in current_results}

        service_results = (
            self.service_retriever.search(current_query or query, top_k=min(self.service_top_k, len(self.service_docs)))
            if self.service_retriever is not None
            else []
        )
        service_rank = {str(item.get("service", "")): int(item["rank"]) for item in service_results}

        candidate_docs: dict[str, dict[str, Any]] = {}
        for item in recency_results + current_results:
            candidate_docs[str(item.get("doc_id"))] = item
        for item in service_results:
            for doc in self.docs_by_service.get(str(item.get("service", "")), []):
                candidate_docs[str(doc.get("doc_id"))] = doc

        # Activate predicted-tool priors only when the current utterance does
        # not explicitly contradict the predicted tool. This prevents the
        # policy and retriever from self-confirming the same wrong route.
        inferred_tool = infer_query_tool(current_query or query)
        tool_filter_active = bool(tool_name) and (
            inferred_tool is None or inferred_tool == tool_name
        )
        if tool_filter_active:
            for doc in self.docs:
                if doc_tool_label(doc) == tool_name:
                    candidate_docs[str(doc.get("doc_id"))] = doc

        local_docs = list(candidate_docs.values())
        recency_local = BM25Retriever(local_docs).search(query, top_k=len(local_docs))
        current_local = BM25Retriever(local_docs).search(current_query or query, top_k=len(local_docs))
        local_rank = {str(item.get("doc_id")): int(item["rank"]) for item in recency_local}
        current_local_rank = {str(item.get("doc_id")): int(item["rank"]) for item in current_local}
        lexical_score = {str(item.get("doc_id")): float(item.get("score", 0.0)) for item in current_local}

        rows: list[dict[str, Any]] = []
        query_tool_override_active = bool(
            inferred_tool and tool_name and inferred_tool != tool_name
        )
        effective_tool = (
            inferred_tool
            if query_tool_override_active
            else (tool_name if tool_filter_active else None)
        )
        shortlisted_services = [str(item.get("service", "")) for item in service_results]
        for doc_id, doc in candidate_docs.items():
            service = str(doc.get("service", ""))
            predicted_tool_match = bool(tool_name) and doc_tool_label(doc) == tool_name
            query_tool_match = bool(effective_tool) and doc_tool_label(doc) == effective_tool
            tool_match = bool(tool_filter_active and predicted_tool_match)
            fused_score = (
                1.20 * _rrf(current_local_rank.get(doc_id))
                + 0.90 * _rrf(local_rank.get(doc_id))
                + 0.55 * _rrf(current_rank.get(doc_id))
                + 0.35 * _rrf(recency_rank.get(doc_id))
                + 0.30 * _rrf(service_rank.get(service))
                + (0.010 if tool_filter_active and tool_match else 0.0)
            )
            features = build_feature_vector(
                doc=doc,
                query=query,
                current_query=current_query or query,
                tool_name=effective_tool,
                fused_score=fused_score,
                recency_rank=local_rank.get(doc_id),
                current_rank=current_local_rank.get(doc_id),
                global_rank=current_rank.get(doc_id),
                service_rank=service_rank.get(service),
            )
            rows.append(
                {
                    **doc,
                    "fused_score": float(fused_score),
                    "lexical_score": lexical_score.get(doc_id, 0.0),
                    "reranker_features": features,
                    "tool_compatible": tool_match,
                    "predicted_tool_match": predicted_tool_match,
                    "query_tool_match": query_tool_match,
                    "query_tool_override_active": query_tool_override_active,
                    "effective_tool": effective_tool,
                    "tool_filter_active": tool_filter_active,
                    "service_shortlist": shortlisted_services,
                    "service_base": service_base(service),
                }
            )
        return rows

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        tool_name: str | None = None,
        current_query: str | None = None,
        use_reranker: bool = True,
        include_features: bool = False,
    ) -> list[dict[str, Any]]:
        top_k = max(1, int(top_k))
        if not self.docs:
            return []
        rows = self._candidate_rows(
            query,
            current_query=str(current_query or query),
            tool_name=tool_name,
        )
        if use_reranker and self.reranker is not None:
            probabilities = self.reranker.score([row["reranker_features"] for row in rows])
            for row, probability in zip(rows, probabilities, strict=True):
                row["reranker_probability"] = probability
                row["final_score"] = 0.85 * probability + 0.15 * row["fused_score"]
            strategy = "recency_tool_filtered_bm25_lr_rerank"
        else:
            for row in rows:
                row["reranker_probability"] = None
                row["final_score"] = row["fused_score"]
            strategy = "recency_tool_filtered_bm25"

        rows.sort(
            key=lambda row: (
                -int(bool(row.get("query_tool_match")))
                if bool(row.get("query_tool_override_active"))
                else 0,
                -int(bool(row.get("tool_compatible")))
                if tool_name and bool(row.get("tool_filter_active"))
                else 0,
                -float(row["final_score"]),
                str(row.get("doc_id", "")),
            )
        )
        output: list[dict[str, Any]] = []
        for rank, row in enumerate(rows[:top_k], start=1):
            clean = dict(row) if include_features else {
                key: value for key, value in row.items() if key != "reranker_features"
            }
            clean.update(
                {
                    "score": float(row["final_score"]),
                    "rank": rank,
                    "retrieval_strategy": strategy,
                }
            )
            output.append(clean)
        return output
