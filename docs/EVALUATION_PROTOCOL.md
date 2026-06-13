# Evaluation Protocol

## Partitions

Internal train, calibration, and ID evaluation use dialogue-group separation
from official SGD train data. Official dev/test services are evaluated as OOD.
Retrieval examples are mined only from internal training rows.

## Model metrics

Report accuracy, macro-F1, class support, ECE, multiclass Brier score, log loss,
clarification recall, premature-action rate, forbidden-tool rate, and execution
errors.

ID tool-routing results are schema-conditioned and are not general
natural-language reasoning accuracy.

## Retrieval metrics

- **Compatible**: semantically equivalent service-family and intent evidence.
- **Strict**: exact service-version and intent provenance.

Report Hit@K, Top-1, and MRR for both. Task claims use compatible metrics;
lineage claims use strict metrics.

## Trajectory and safety metrics

Report groundedness, trajectory success, safety block rate, false-block rate,
trace completeness, latency percentiles, and PASS/REVIEW/BLOCK distribution.

## Replay

Compare stable fields: tools, retrieved IDs, execution status, retries, release
decision, failure attribution, rounded scores, and fingerprints. Exclude run
IDs, timestamps, row IDs, and absolute latency.

## Ablations

Compare full input with utterance/context only, schema descriptions only,
metadata names only, slots/risk only, and no-oracle structured input. Use both
masked Transformer inference and separately trained linear baselines.

## Claim policy

Every public claim must identify the dataset/split, metric definition,
comparison baseline, environment when relevant, and known limitations.
