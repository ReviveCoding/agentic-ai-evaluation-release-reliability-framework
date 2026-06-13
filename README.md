# Agentic AI Evaluation & Release Reliability Framework

<!-- public-release-badges -->
[![CI](https://github.com/ReviveCoding/agentic-ai-evaluation-release-reliability-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/ReviveCoding/agentic-ai-evaluation-release-reliability-framework/actions/workflows/ci.yml)
[![CodeQL](https://github.com/ReviveCoding/agentic-ai-evaluation-release-reliability-framework/actions/workflows/codeql.yml/badge.svg)](https://github.com/ReviveCoding/agentic-ai-evaluation-release-reliability-framework/actions/workflows/codeql.yml)
[![Release](https://img.shields.io/github/v/release/ReviveCoding/agentic-ai-evaluation-release-reliability-framework)](https://github.com/ReviveCoding/agentic-ai-evaluation-release-reliability-framework/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)


**GPU-Trained Tool Policy, Retrieval Hardening, Deterministic Replay, and Release Gates**

An end-to-end framework for training, executing, evaluating, replaying, and release-gating tool-using AI workflows.

The framework trains a Transformer-based tool-routing policy on public Schema-Guided Dialogue data, executes multi-step assistant workflows, retrieves supporting evidence, records complete trace lineage, attributes failures, and produces automated PASS, REVIEW, or BLOCK release decisions.

## Highlights

- GPU-trained DistilBERT tool-policy model using CUDA and BF16
- Dialogue-group-held-out in-domain evaluation and official SGD dev/test OOD benchmarks
- Recency-aware hierarchical BM25 retrieval with lightweight learned reranking
- Artifact-independent self-confirmation protection when model predictions conflict with current user intent
- Compatible and strict retrieval metrics for task success and exact provenance
- Stable-field deterministic replay with model and dataset fingerprinting
- Failure attribution, latency analysis, safety metrics, and automated release gates
- FastAPI and CLI interfaces, Docker support, GitHub Actions, and 70 passing regression tests

## Final validated results

| Metric | Result |
|---|---:|
| In-domain tool-routing accuracy | 99.47% |
| In-domain compatible retrieval Hit@5 | 97.73% |
| In-domain compatible retrieval Top-1 | 64.29% |
| In-domain groundedness | 80.74% |
| In-domain trajectory success | 80.74% |
| In-domain release PASS rate | 56.92% |
| OOD dev groundedness | 75.69% |
| OOD dev release PASS rate | 63.49% |
| OOD test groundedness | 71.11% |
| OOD test release PASS rate | 53.39% |
| Stable-field replay verification | 100.00% |
| Regression tests | 70 passed |

## Architecture

```text
Public SGD data -> validation and stratified split -> GPU tool policy
-> agentic execution -> recency-aware retrieval and reranking
-> evaluator registry -> trace store and replay -> PASS / REVIEW / BLOCK
```

Full architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## Quick start

### Lightweight CPU smoke test

```powershell
python scripts_build_dataset.py --use-sample
python scripts/02_train_model.py --backend sklearn
python scripts/03_run_evaluation.py
python scripts/04_replay_failures.py
python scripts/05_export_reports.py
python -m pytest -q
```

### Public SGD and GPU path

```powershell
python scripts/00_validate_raw_dataset.py --raw-dir "C:/path/to/dstc8-schema-guided-dialogue"
python scripts/12_build_stratified_sgd.py --raw-dir "C:/path/to/dstc8-schema-guided-dialogue" --max-dialogues 500 --max-ood-dialogues 250
python scripts/02_train_model.py --backend transformer --model-name distilbert-base-uncased --epochs 2 --batch-size 16 --gradient-accumulation-steps 2 --gradient-checkpointing
python scripts/03_run_evaluation.py
python scripts/04_replay_failures.py
python scripts/13_evaluate_ood.py
python scripts/19_evaluate_retrieval_top1.py
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Final results](docs/RESULTS.md)
- [Model card](docs/MODEL_CARD.md)
- [Limitations and claim boundaries](docs/LIMITATIONS.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Resume and interview claims](docs/RESUME_CLAIMS.md)
- [Release checklist](docs/RELEASE_CHECKLIST.md)
- [Changelog](CHANGELOG.md)

## Claim boundary

This is a local and GitHub-runnable, public-data, production-style evaluation framework. It does not use Apple data, private user data, production Siri systems, or real customer workflows. In-domain results are schema-conditioned and should not be interpreted as general agent reasoning performance.

Compatible retrieval and strict provenance are reported separately. Replay verification is scoped to the same model, dataset, software, and hardware configuration.

## Release

The final functional version is frozen as **v1.0.0**. Future changes should be documentation, compatibility, or bug-fix updates unless a new major evaluation objective is explicitly introduced.


## Five-minute reviewer tour

1. Read the [architecture](docs/ARCHITECTURE.md).
2. Review the [validated results](docs/RESULTS.md).
3. Check the [evaluation protocol](docs/EVALUATION_PROTOCOL.md).
4. Inspect the [limitations](docs/LIMITATIONS.md) and [threat model](docs/THREAT_MODEL.md).
5. Follow the [repository map](docs/REPOSITORY_MAP.md) to key tests.

## Community and security

- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Support](SUPPORT.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
- [GitHub settings checklist](docs/GITHUB_SETTINGS_CHECKLIST.md)
- [Apache License 2.0](LICENSE)
