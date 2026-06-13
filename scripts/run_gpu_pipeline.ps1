# Windows PowerShell helper for local GPU-capable training.
# Run from the project root after creating and activating your venv.

$env:PYTHONPATH = "src"

python scripts/01_build_dataset.py --use-sample
python scripts/02_train_model.py --backend transformer --model-name distilbert-base-uncased --epochs 2 --batch-size 16
python scripts/03_run_evaluation.py
python scripts/04_replay_failures.py
python scripts/05_export_reports.py
pytest -q
