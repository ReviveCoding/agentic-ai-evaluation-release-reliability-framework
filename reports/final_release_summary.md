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



# Final Results

## Validated release

- Release: **v1.0.0**
- Final hardening status: **PASS_MEANINGFUL_GAIN**
- Regression tests passed: **70**
- Replay verification: **100.00%** across **246** failed or reviewed runs

## In-domain evaluation

| Metric | Result |
|---|---:|
| Tool-routing accuracy | 99.47% |
| Compatible retrieval Hit@5 | 97.73% |
| Strict retrieval Hit@5 | 36.04% |
| Compatible retrieval Top-1 | 64.29% |
| Strict retrieval Top-1 | 15.91% |
| Compatible MRR | 0.7883 |
| Groundedness | 80.74% |
| Trajectory success | 80.74% |
| Release PASS rate | 56.92% |
| Trace completeness | 100.00% |
| Safety block rate | 100.00% |
| Unsafe completion rate | 0.00% |

## OOD evaluation

| Metric | Official dev | Official test |
|---|---:|---:|
| Tool accuracy | 90.63% | 95.94% |
| Compatible Hit@5 | 81.47% | 76.10% |
| Compatible Top-1 | 66.60% | 51.30% |
| Groundedness | 75.69% | 71.11% |
| Trajectory success | 75.69% | 71.11% |
| Release PASS rate | 63.49% | 53.39% |
| Safety block rate | 100.00% | 90.10% |

## Hardening impact

| Improvement | Change |
|---|---:|
| ID compatible Top-1 | +28.2 pp |
| ID groundedness | +15.8 pp |
| ID trajectory success | +15.8 pp |
| ID release PASS rate | +15.8 pp |
| Replay verification | 100.00% final |

## Interpretation

The tool policy is strong on the dialogue-group-held-out in-domain split and remains effective on official OOD services. End-to-end success is lower than classifier accuracy because the framework requires correct tool selection, compatible evidence, safe execution, and complete trajectory behavior simultaneously.

Compatible and strict retrieval metrics answer different questions. Compatible metrics accept semantically equivalent service-family evidence for execution, while strict metrics require exact service-version and intent lineage for provenance.
