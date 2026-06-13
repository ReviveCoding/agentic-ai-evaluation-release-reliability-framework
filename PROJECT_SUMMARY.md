# Agentic AI Evaluation & Release Reliability Framework

**GPU-Trained Tool Policy, Retrieval Hardening, Deterministic Replay, and Release Gates**

This project is an end-to-end framework for training, executing, evaluating, replaying, and release-gating tool-using AI workflows. It combines a GPU-trained Transformer policy, public SGD data, hierarchical retrieval, learned reranking, trace lineage, failure attribution, stable-field replay, and automated release decisions.

## Final release snapshot

- Version: v1.0.0
- In-domain tool accuracy: 99.47%
- In-domain compatible retrieval Top-1: 64.29%
- In-domain grounded trajectory success: 80.74%
- OOD dev/test groundedness: 75.69% / 71.11%
- Replay verification: 100.00% across 246 failed or reviewed runs
- Regression tests: 70 passed
- Final hardening status: PASS_MEANINGFUL_GAIN

See `README.md` and the `docs/` directory for architecture, results, reproducibility, limitations, model details, and resume-safe claims.
