# Monte Carlo End-to-End Validation Report

- Seed: `20260611`
- Replications: `4`
- Scenarios per replication: `40`
- Total evaluated runs: `640`

## Variant summary

| Variant | Tool accuracy | Trajectory success | Groundedness | Confidence | Clarification recall | Retry recovery | Error rate | PASS rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid | 0.919 | 0.787 | 0.750 | 0.699 | 0.958 | 0.905 | 0.008 | 0.562 |
| hybrid_no_retry | 0.919 | 0.762 | 0.691 | 0.699 | 0.958 | N/A | 0.091 | 0.512 |
| learned | 0.919 | 0.787 | 0.750 | 0.782 | 0.958 | 0.905 | 0.008 | 0.562 |
| rule | 0.838 | 0.700 | 0.672 | 0.000 | 0.750 | 0.889 | 0.009 | 0.469 |

## Difficulty slices

### hybrid

| Difficulty | N | Tool accuracy | Trajectory success | Groundedness | Timeout rate | PASS | REVIEW | BLOCK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| easy | 40 | 1.000 | 0.850 | 0.850 | 0.016 | 25 | 15 | 0 |
| medium | 40 | 0.925 | 0.800 | 0.800 | 0.032 | 24 | 16 | 0 |
| hard | 40 | 0.825 | 0.700 | 0.688 | 0.000 | 21 | 18 | 1 |
| stressed | 40 | 0.925 | 0.800 | 0.662 | 0.117 | 20 | 20 | 0 |

### hybrid_no_retry

| Difficulty | N | Tool accuracy | Trajectory success | Groundedness | Timeout rate | PASS | REVIEW | BLOCK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| easy | 40 | 1.000 | 0.850 | 0.825 | 0.017 | 24 | 16 | 0 |
| medium | 40 | 0.925 | 0.750 | 0.750 | 0.034 | 24 | 16 | 0 |
| hard | 40 | 0.825 | 0.675 | 0.613 | 0.000 | 18 | 21 | 1 |
| stressed | 40 | 0.925 | 0.775 | 0.575 | 0.086 | 16 | 24 | 0 |

### learned

| Difficulty | N | Tool accuracy | Trajectory success | Groundedness | Timeout rate | PASS | REVIEW | BLOCK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| easy | 40 | 1.000 | 0.850 | 0.850 | 0.016 | 25 | 15 | 0 |
| medium | 40 | 0.925 | 0.800 | 0.800 | 0.032 | 24 | 16 | 0 |
| hard | 40 | 0.825 | 0.700 | 0.688 | 0.000 | 21 | 18 | 1 |
| stressed | 40 | 0.925 | 0.800 | 0.662 | 0.117 | 20 | 20 | 0 |

### rule

| Difficulty | N | Tool accuracy | Trajectory success | Groundedness | Timeout rate | PASS | REVIEW | BLOCK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| easy | 40 | 1.000 | 0.850 | 0.850 | 0.016 | 25 | 15 | 0 |
| medium | 40 | 0.925 | 0.825 | 0.825 | 0.032 | 23 | 17 | 0 |
| hard | 40 | 0.750 | 0.625 | 0.613 | 0.000 | 17 | 23 | 0 |
| stressed | 40 | 0.675 | 0.500 | 0.400 | 0.108 | 10 | 30 | 0 |

## Tool slices

### hybrid

| Expected first tool | N | Tool accuracy | Trajectory success | Groundedness | PASS | REVIEW | BLOCK |
|---|---:|---:|---:|---:|---:|---:|---:|
| ask_clarification | 24 | 0.958 | 0.958 | 0.875 | 21 | 3 | 0 |
| calendar_lookup | 16 | 0.938 | 0.875 | 0.875 | 12 | 4 | 0 |
| calendar_write | 16 | 0.750 | 0.750 | 0.688 | 0 | 16 | 0 |
| media_search | 16 | 1.000 | 1.000 | 0.969 | 14 | 2 | 0 |
| safety_check | 16 | 0.938 | 0.938 | 0.875 | 0 | 15 | 1 |
| search_docs | 28 | 0.929 | 0.893 | 0.875 | 24 | 4 | 0 |
| search_places | 28 | 0.857 | 0.214 | 0.214 | 6 | 22 | 0 |
| weather_lookup | 16 | 1.000 | 0.938 | 0.875 | 13 | 3 | 0 |

### hybrid_no_retry

| Expected first tool | N | Tool accuracy | Trajectory success | Groundedness | PASS | REVIEW | BLOCK |
|---|---:|---:|---:|---:|---:|---:|---:|
| ask_clarification | 24 | 0.958 | 0.958 | 0.875 | 21 | 3 | 0 |
| calendar_lookup | 16 | 0.938 | 0.812 | 0.719 | 10 | 6 | 0 |
| calendar_write | 16 | 0.750 | 0.750 | 0.562 | 0 | 16 | 0 |
| media_search | 16 | 1.000 | 0.938 | 0.875 | 13 | 3 | 0 |
| safety_check | 16 | 0.938 | 0.938 | 0.875 | 0 | 15 | 1 |
| search_docs | 28 | 0.929 | 0.893 | 0.839 | 22 | 6 | 0 |
| search_places | 28 | 0.857 | 0.214 | 0.196 | 5 | 23 | 0 |
| weather_lookup | 16 | 1.000 | 0.812 | 0.750 | 11 | 5 | 0 |

### learned

| Expected first tool | N | Tool accuracy | Trajectory success | Groundedness | PASS | REVIEW | BLOCK |
|---|---:|---:|---:|---:|---:|---:|---:|
| ask_clarification | 24 | 0.958 | 0.958 | 0.875 | 21 | 3 | 0 |
| calendar_lookup | 16 | 0.938 | 0.875 | 0.875 | 12 | 4 | 0 |
| calendar_write | 16 | 0.750 | 0.750 | 0.688 | 0 | 16 | 0 |
| media_search | 16 | 1.000 | 1.000 | 0.969 | 14 | 2 | 0 |
| safety_check | 16 | 0.938 | 0.938 | 0.875 | 0 | 15 | 1 |
| search_docs | 28 | 0.929 | 0.893 | 0.875 | 24 | 4 | 0 |
| search_places | 28 | 0.857 | 0.214 | 0.214 | 6 | 22 | 0 |
| weather_lookup | 16 | 1.000 | 0.938 | 0.875 | 13 | 3 | 0 |

### rule

| Expected first tool | N | Tool accuracy | Trajectory success | Groundedness | PASS | REVIEW | BLOCK |
|---|---:|---:|---:|---:|---:|---:|---:|
| ask_clarification | 24 | 0.750 | 0.750 | 0.708 | 17 | 7 | 0 |
| calendar_lookup | 16 | 0.812 | 0.750 | 0.750 | 10 | 6 | 0 |
| calendar_write | 16 | 0.875 | 0.875 | 0.812 | 0 | 16 | 0 |
| media_search | 16 | 0.625 | 0.625 | 0.625 | 9 | 7 | 0 |
| safety_check | 16 | 1.000 | 1.000 | 0.938 | 0 | 16 | 0 |
| search_docs | 28 | 0.893 | 0.821 | 0.804 | 22 | 6 | 0 |
| search_places | 28 | 0.857 | 0.214 | 0.214 | 6 | 22 | 0 |
| weather_lookup | 16 | 0.875 | 0.812 | 0.750 | 11 | 5 | 0 |

## Paired replication differences

### hybrid_minus_rule

- `tool_accuracy` mean difference: `0.0812` (95% bootstrap CI `0.0375` to `0.1250`)
- `trajectory_success` mean difference: `0.0875` (95% bootstrap CI `0.0625` to `0.1125`)
- `groundedness` mean difference: `0.0781` (95% bootstrap CI `0.0563` to `0.1094`)

### hybrid_minus_learned

- `tool_accuracy` mean difference: `0.0000` (95% bootstrap CI `0.0000` to `0.0000`)
- `trajectory_success` mean difference: `0.0000` (95% bootstrap CI `0.0000` to `0.0000`)
- `groundedness` mean difference: `0.0000` (95% bootstrap CI `0.0000` to `0.0000`)

### hybrid_minus_hybrid_no_retry

- `tool_accuracy` mean difference: `0.0000` (95% bootstrap CI `0.0000` to `0.0000`)
- `trajectory_success` mean difference: `0.0250` (95% bootstrap CI `0.0250` to `0.0250`)
- `groundedness` mean difference: `0.0594` (95% bootstrap CI `0.0438` to `0.0750`)

## Sanity checks

- PASS: `hybrid_stress_degrades_or_matches` - `{"easy": 0.85, "stressed": 0.8}`
- PASS: `hybrid_trace_complete` - `1.0`
- PASS: `hybrid_no_retry_stress_degrades_or_matches` - `{"easy": 0.85, "stressed": 0.775}`
- PASS: `hybrid_no_retry_trace_complete` - `1.0`
- PASS: `learned_stress_degrades_or_matches` - `{"easy": 0.85, "stressed": 0.8}`
- PASS: `learned_trace_complete` - `1.0`
- PASS: `rule_stress_degrades_or_matches` - `{"easy": 0.85, "stressed": 0.5}`
- PASS: `rule_trace_complete` - `1.0`
- PASS: `hybrid_matches_best_single_policy` - `{"hybrid": 0.91875, "best_single": 0.91875}`
- PASS: `retry_reduces_execution_errors` - `{"with_retry": 0.00847457627118644, "without_retry": 0.09051724137931035}`
- PASS: `retry_improves_or_preserves_groundedness` - `{"with_retry": 0.75, "without_retry": 0.690625}`
- PASS: `hybrid_retrieval_hit_at_k_coverage_sufficient` - `{"coverage": 0.64375, "minimum": 0.25}`
- PASS: `hybrid_clarification_recall_coverage_sufficient` - `{"coverage": 0.15, "minimum": 0.05}`
- PASS: `hybrid_safety_block_rate_coverage_sufficient` - `{"coverage": 0.1, "minimum": 0.05}`
- PASS: `hybrid_no_unsafe_completion` - `0.0`

## Interpretation

The simulation deliberately combines metadata noise, utterance perturbations, transient errors, timeouts, corrupted evidence, and long-tail latency. A reasonable framework should perform better on easy than stressed slices, preserve trace completeness, recover a material share of retryable failures, and avoid treating wrong-tool evidence as grounded.
