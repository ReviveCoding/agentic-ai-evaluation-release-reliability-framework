# Changelog

All notable changes to this project are documented in this file.

## [1.0.0] - 2026-06-12

### Added

- Dialogue-group-stratified public SGD training, calibration, ID evaluation, and official OOD evaluation
- CUDA/BF16 Transformer tool-policy path
- Recency-aware hierarchical BM25 retrieval
- Lightweight logistic-regression reranker
- Artifact-independent policy self-confirmation guard
- Compatible and strict retrieval metrics
- Stable-field deterministic replay
- Input ablation reporting
- Final architecture, results, model card, limitations, reproducibility, and resume claim documentation

### Validated

- 99.47% in-domain tool accuracy
- 80.74% in-domain trajectory success
- 75.69% / 71.11% OOD dev/test groundedness
- 100.00% replay verification across 246 failed or reviewed runs
- 70 passing regression tests

### Changed

- Project version promoted from the development series to v1.0.0
- Metadata-based rule results are documented as oracle-style references rather than fair learned baselines
- Final feature version frozen; subsequent work should focus on documentation, compatibility, and bug fixes
