# Limitations and Claim Boundaries

## Scope

This is a local and GitHub-runnable, public-data, production-style evaluation framework. It does not use Apple data, private user data, production Siri systems, or real customer workflows.

## Model interpretation

The tool-routing task is schema-conditioned and uses service descriptions, intent descriptions, slot state, risk indicators, user utterances, and dialogue context. Strong in-domain classification should not be interpreted as general conversational reasoning or universal agent intelligence.

## Retrieval interpretation

- Compatible retrieval accepts semantically equivalent service-family evidence and is the primary task-execution metric.
- Strict retrieval requires exact service-version and intent provenance and is reported separately.
- A higher compatible score does not imply perfect exact lineage.

## Replay interpretation

The final replay verification rate is 100.00% across 246 failed or reviewed runs. This is stable-field verification under the same model, dataset, software, and hardware configuration. It does not claim universal cross-platform or cross-version determinism.

## Deployment boundary

The project demonstrates engineering patterns for evaluation infrastructure, observability, and release readiness. It has not been deployed to a production assistant or validated against live customer traffic.

## Remaining limitations

- Exact strict-provenance retrieval is materially lower than compatible retrieval.
- Official OOD test safety blocking is not perfect.
- Latency measurements are local-machine measurements, not distributed serving benchmarks.
- The metadata-based rule result is an oracle-style reference, not a fair utterance-only ML baseline.
- The optional LLM judge and dense retriever are extension hooks, not requirements of the validated default path.
