# Architecture

## System objective

The framework evaluates tool-using AI workflows as an integrated system rather than scoring only a classifier. It connects public-data model training, multi-step execution, evidence retrieval, trace lineage, replay, failure attribution, and release decisions.

## End-to-end flow

```text
Public Schema-Guided Dialogue data
        |
        v
Schema validation and mapping audit
        |
        v
Dialogue-group stratified split
        |-- model train
        |-- calibration
        |-- in-domain evaluation
        |-- official dev OOD
        `-- official test OOD
        |
        v
GPU Transformer tool policy
        |
        v
Agentic execution engine
        |-- planning and tool selection
        |-- async tool execution
        |-- retry and timeout handling
        `-- safety and clarification logic
        |
        v
Recency-aware retrieval
        |-- hierarchical BM25
        |-- tool-compatible candidate generation
        |-- lightweight logistic-regression reranking
        `-- query/policy self-confirmation guard
        |
        v
Evaluator registry
        |-- tool accuracy
        |-- retrieval and evidence compatibility
        |-- groundedness and trajectory success
        |-- safety and clarification
        |-- latency and trace completeness
        `-- failure attribution
        |
        v
Trace store and stable-field replay
        |
        v
PASS / REVIEW / BLOCK release gate
```

## Major components

| Component | Responsibility |
|---|---|
| Data pipeline | Validates SGD data, maps services and intents to framework tools, and prevents dialogue overlap across splits. |
| Tool-policy model | DistilBERT sequence classifier trained with CUDA and BF16 when available. |
| Execution engine | Runs scenarios, tools, retries, timeouts, safety checks, and trace persistence. |
| Retrieval stack | Uses recency-weighted queries, hierarchical BM25, learned reranking, and self-confirmation protection. |
| Evaluators | Score policy, retrieval, grounding, trajectory, safety, latency, and completeness. |
| Replay engine | Re-executes failed or reviewed runs and compares stable signatures. |
| Release gate | Produces PASS, REVIEW, or BLOCK decisions from evaluator outputs. |
| Interfaces | Exposes CLI, FastAPI, reports, Docker, and GitHub Actions paths. |

## Final design decisions

- The default framework remains lean: no Kubernetes, LoRA, full dashboard, or mandatory LLM judge.
- Compatible retrieval measures task-useful evidence; strict retrieval separately measures exact service-version provenance.
- Query evidence can override an incorrect policy prior only when current user intent explicitly conflicts with the predicted tool.
- Replay excludes timestamps, wall-clock latency, run IDs, and database row IDs from deterministic equality.
