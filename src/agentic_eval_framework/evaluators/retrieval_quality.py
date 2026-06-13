from __future__ import annotations


def retrieval_metrics(observations: list[dict]) -> dict[str, float]:
    retrieval_observations = [obs for obs in observations if "retrieved_doc_ids" in obs]
    if not retrieval_observations:
        return {
            "retrieval_applicable": 0.0,
            "retrieval_hit_at_k": 0.0,
            "retrieval_strict_hit_at_k": 0.0,
            "retrieval_top1_accuracy": 0.0,
            "retrieval_strict_top1_accuracy": 0.0,
            "retrieval_reciprocal_rank": 0.0,
            "retrieval_strict_reciprocal_rank": 0.0,
        }

    compatible_hits: list[float] = []
    strict_hits: list[float] = []
    compatible_top1: list[float] = []
    strict_top1: list[float] = []
    compatible_rr: list[float] = []
    strict_rr: list[float] = []

    for obs in retrieval_observations:
        target = str(obs.get("retrieval_target_doc_id", ""))
        compatible = set(obs.get("retrieval_compatible_doc_ids") or ([target] if target else []))
        ids = [str(item) for item in (obs.get("retrieved_doc_ids") or [])]
        compatible_ranks = [idx + 1 for idx, doc_id in enumerate(ids) if doc_id in compatible]
        strict_rank = ids.index(target) + 1 if target in ids else None

        compatible_hits.append(float(bool(compatible_ranks)))
        strict_hits.append(float(strict_rank is not None))
        compatible_top1.append(float(bool(ids) and ids[0] in compatible))
        strict_top1.append(float(bool(ids) and ids[0] == target))
        compatible_rr.append(1.0 / min(compatible_ranks) if compatible_ranks else 0.0)
        strict_rr.append(1.0 / strict_rank if strict_rank is not None else 0.0)

    count = len(retrieval_observations)
    return {
        "retrieval_applicable": 1.0,
        "retrieval_hit_at_k": sum(compatible_hits) / count,
        "retrieval_strict_hit_at_k": sum(strict_hits) / count,
        "retrieval_top1_accuracy": sum(compatible_top1) / count,
        "retrieval_strict_top1_accuracy": sum(strict_top1) / count,
        "retrieval_reciprocal_rank": sum(compatible_rr) / count,
        "retrieval_strict_reciprocal_rank": sum(strict_rr) / count,
    }
