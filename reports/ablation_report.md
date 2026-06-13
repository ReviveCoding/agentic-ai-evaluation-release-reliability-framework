# ID Input Ablation Report

This report separates model capability from structured-metadata shortcuts.

## Same Transformer, masked inputs

| Input mode | Accuracy | Macro-F1 | Drop vs full | Mean confidence |
|---|---:|---:|---:|---:|
| full | 1.0000 | 1.0000 | 0.0000 | 0.9416 |
| utterance_context_only | 0.5762 | 0.4144 | 0.5856 | 0.7288 |
| schema_descriptions_only | 0.6357 | 0.5854 | 0.4146 | 0.8431 |
| metadata_names_only | 0.5254 | 0.3653 | 0.6347 | 0.6289 |
| slots_risk_only | 0.8739 | 0.5978 | 0.4022 | 0.8623 |
| no_oracle_structured | 0.6340 | 0.5683 | 0.4317 | 0.8885 |

## Independently retrained linear baselines

| Input mode | Accuracy | Macro-F1 | Drop vs full |
|---|---:|---:|---:|
| full | 0.8249 | 0.8449 | 0.0000 |
| utterance_context_only | 0.6637 | 0.5802 | 0.2647 |
| schema_descriptions_only | 0.7776 | 0.7719 | 0.0730 |
| metadata_names_only | 0.7776 | 0.7719 | 0.0730 |
| slots_risk_only | 0.8336 | 0.6800 | 0.1648 |
| no_oracle_structured | 0.7898 | 0.8089 | 0.0360 |

## Metadata lookup oracle

- Accuracy: 0.9264
- Macro-F1: 0.9495
- Key coverage: 0.9264

## Permutation sensitivity

| Permuted feature group | Accuracy | Macro-F1 | Macro-F1 drop |
|---|---:|---:|---:|
| service_intent | 1.0000 | 1.0000 | 0.0000 |
| schema_descriptions | 0.9177 | 0.6800 | 0.3200 |
| slots_risk | 0.5499 | 0.5502 | 0.4498 |
| utterance_context | 0.9982 | 0.9969 | 0.0031 |
