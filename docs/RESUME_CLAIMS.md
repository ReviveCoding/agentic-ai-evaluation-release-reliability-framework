# Resume and Interview Claims

## Recommended project title

**Agentic AI Evaluation & Release Reliability Framework**
Python, PyTorch, Transformers, CUDA, FastAPI, SQLite, BM25, scikit-learn, Docker, GitHub Actions

## General resume version

- Built a local and GitHub-runnable evaluation framework for tool-using AI workflows, integrating GPU-trained tool routing, multi-step execution traces, evaluator plugins, failure attribution, deterministic replay, and automated PASS/REVIEW/BLOCK release gates.
- Trained a BF16 DistilBERT tool-policy model on a dialogue-group-stratified public SGD dataset, achieving 99.5% in-domain tool-routing accuracy while separately benchmarking official dev/test OOD services.
- Improved compatible retrieval Top-1 by +28.2 pp and grounded trajectory success by +15.8 pp using recency-aware hierarchical BM25, lightweight learned reranking, and an artifact-independent self-confirmation guard; verified 100.00% stable-field replay across 246 failed or reviewed runs.

## Systems / Evaluation Engineering version

- Built an end-to-end agentic evaluation execution framework with asynchronous workflow execution, trace lineage, retry and timeout handling, evaluator registries, deterministic replay, and automated release gates.
- Implemented FastAPI and CLI interfaces, SQLite trace persistence, Docker and GitHub Actions workflows, latency reporting, failure attribution, and 70 passing regression tests for local and CI reproducibility.
- Added recency-aware retrieval and learned reranking, increasing grounded trajectory success from 64.97% to 80.74% while preserving 100.00% stable-field replay verification.

## Software / Agentic Evaluation version

- Developed reusable tooling for measuring multi-step AI workflow quality across tool selection, evidence retrieval, grounding, safety, latency, and trajectory completion.
- Built in-domain and OOD evaluation pipelines, automated reports, replay diagnostics, and PASS/REVIEW/BLOCK quality gates with complete trace coverage.
- Hardened retrieval against stale context and policy self-confirmation, increasing the in-domain release PASS rate by +15.8 pp without retraining the underlying Transformer policy.

## Applied Research / Proactive Intelligence version

- Trained and evaluated a schema-conditioned Transformer policy for tool-using workflows using public SGD data, dialogue-group-held-out evaluation, calibration, and official unseen-service OOD benchmarks.
- Conducted full, utterance-only, schema-only, metadata-only, and structured-state ablations to separate language understanding from schema-conditioned routing and identify shortcut dependencies.
- Improved OOD groundedness through hierarchical retrieval, recency-aware context weighting, learned reranking, and failure analysis of policy, retrieval, and trajectory errors.

## Thirty-second interview explanation

I built an end-to-end agentic evaluation framework rather than only training a classifier. It trains a GPU-based tool-routing model on public Schema-Guided Dialogue data, executes multi-step workflows, retrieves supporting evidence, records full traces, attributes failures, and makes automated release decisions. After identifying retrieval as the main bottleneck, I added recency-aware retrieval, a lightweight reranker, and a self-confirmation guard, improving grounded trajectory success by +15.8 pp while preserving 100.00% stable-field replay verification.

## Safe claim boundaries

Use:
- local and GitHub-runnable
- public SGD-derived data
- production-style evaluation framework
- dialogue-group-held-out in-domain evaluation
- official dev/test OOD evaluation
- stable-field replay under the same environment

Avoid:
- production deployed
- Apple-scale or Siri platform
- general agent reasoning accuracy of 99 percent
- universally deterministic replay
- human-level or production-ready agent performance
