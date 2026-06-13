from __future__ import annotations

import numpy as np


def latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
    arr = np.asarray(latencies_ms, dtype=float)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
    }
