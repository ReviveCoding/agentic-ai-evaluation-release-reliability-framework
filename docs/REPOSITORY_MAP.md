# Repository Map

| Path | Purpose |
|---|---|
| `src/agentic_eval_framework/` | Installable framework package |
| `scripts/` | Data, training, evaluation, replay, reporting, and audit entry points |
| `configs/` | Dataset, tool, evaluator, model, and release-gate configuration |
| `tests/` | Regression, safety, retrieval, replay, and packaging tests |
| `data/raw/sgd/` | Tiny checked-in smoke fixture only |
| `data/processed/` | Generated locally and ignored |
| `docs/` | Architecture, results, protocol, limitations, and governance |
| `reports/` | Selected release evidence |
| `.github/workflows/` | CI, CodeQL, real-data validation, and release checks |

## Reviewer path

1. `README.md`
2. `docs/ARCHITECTURE.md`
3. `docs/RESULTS.md`
4. `docs/EVALUATION_PROTOCOL.md`
5. `docs/LIMITATIONS.md`
6. `docs/THREAT_MODEL.md`
7. key retrieval and replay regression tests
