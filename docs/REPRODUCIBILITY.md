# Reproducibility

## Validated environment

- Release: v1.0.0
- Python: 3.11.9
- Platform used to generate documentation: Windows-10-10.0.26200-SP0
- Training path: CUDA-enabled PyTorch with BF16 on an NVIDIA RTX 4090 Laptop GPU
- Regression tests: 70 passed, 0 failed, 0 errors

## Core commands

```powershell
python scripts/00_validate_raw_dataset.py --raw-dir "C:/path/to/dstc8-schema-guided-dialogue"
python scripts/12_build_stratified_sgd.py --raw-dir "C:/path/to/dstc8-schema-guided-dialogue" --max-dialogues 500 --max-ood-dialogues 250
python scripts/02_train_model.py --backend transformer --model-name distilbert-base-uncased --epochs 2 --batch-size 16 --gradient-accumulation-steps 2 --gradient-checkpointing
python scripts/03_run_evaluation.py
python scripts/04_replay_failures.py
python scripts/13_evaluate_ood.py
python scripts/19_evaluate_retrieval_top1.py
python -m pytest -q
python scripts_project_audit.py
```

## Determinism controls

- Fixed dataset and scenario seeds
- Dialogue-group split boundaries
- Model and dataset fingerprints in traces
- Deterministic retrieval tie-breaking
- Stable-field replay signatures
- Dynamic timestamps, absolute latency, run IDs, and database IDs excluded from equality

## Expected artifacts

- `models/tool_policy/model_manifest.json`
- `models/tool_policy/metrics.json`
- `models/retrieval_reranker.joblib`
- `outputs/evaluation_results.json`
- `outputs/ood_summary.json`
- `outputs/replay_failures.json`
- `outputs/top1_hardening_comparison.json`
- `reports/final_release_summary.md`
- `reports/final_release_manifest.json`
