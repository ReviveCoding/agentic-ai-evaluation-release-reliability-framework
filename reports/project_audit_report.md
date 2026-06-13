# Project Audit Report

Status: **PASS**

## Required path check
- All required source/config/test/devops paths are present.

## Expected reports
- All expected reports are present.

## Local/GitHub runnable commands
- `python scripts/00_validate_raw_dataset.py --raw-dir /path/to/sgd`
- `python scripts/11_run_real_data_pipeline.py --raw-dir /path/to/sgd --backend sklearn`
- `python scripts/14_run_stratified_real_data_pipeline.py --raw-dir /path/to/sgd --backend sklearn --max-dialogues 250`
- `python scripts/01_build_dataset.py --use-sample`
- `python scripts/02_train_model.py --backend sklearn`
- `python scripts/03_run_evaluation.py`
- `python scripts/04_replay_failures.py`
- `python scripts/05_export_reports.py`
- `python scripts/09_run_monte_carlo.py --replications 4 --scenarios-per-replication 40`
- `python scripts/10_verify_package.py`
- `pytest -q`
- `make release-check`

## Deployment/reproducibility signals
- github_actions_present: True
- dockerfile_present: True
- dockerignore_present: True
- gpu_training_path_present: True
- cache_dirs_present: False
- version_consistent: True
- retrieval_integration_present: True
- ci_monte_carlo_present: True
- real_data_ci_present: True
