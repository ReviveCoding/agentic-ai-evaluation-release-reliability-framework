# Model Card

## Model

- Base model: `distilbert-base-uncased`
- Task: schema-conditioned tool routing
- Backend: transformer
- Release: v1.0.0
- Training precision: BF16 when CUDA is available

## Inputs

- Current user utterance
- Dialogue context
- Service and intent descriptions
- Known, required, and missing slots
- Risk flags

## Outputs

The classifier predicts the first framework action, including clarification, retrieval, search, calendar, weather, media, and safety-related routes.

## Evaluation protocol

- Dialogue-group-held-out in-domain evaluation
- Independent calibration split
- Official SGD dev and test as OOD benchmarks
- End-to-end agentic evaluation after classification

## Final performance

| Metric | Result |
|---|---:|
| In-domain tool accuracy | 99.47% |
| OOD dev tool accuracy | 90.63% |
| OOD test tool accuracy | 95.94% |
| Mean model confidence, ID | 93.26% |
| Calibration ECE | 0.0585 |

## Ablation interpretation

The validated ablation artifact is available at `outputs/ablation_summary.json`; it separates full, utterance/context-only, schema-only, metadata-only, and structured-state inputs.

The strongest full-input score should be interpreted as structured, schema-conditioned routing rather than pure utterance understanding. The model is only one component of the final framework; release decisions additionally depend on retrieval, grounding, safety, execution, and trajectory completion.
