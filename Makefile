PYTHON ?= python
ENV := PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
RAW_DIR ?= data/raw/sgd
BACKEND ?= sklearn
MODEL_NAME ?= distilbert-base-uncased
MAX_DIALOGUES ?=
MAX_DIALOGUES_ARG := $(if $(MAX_DIALOGUES),--max-dialogues $(MAX_DIALOGUES),)

validate-raw:
	$(ENV) $(PYTHON) scripts/00_validate_raw_dataset.py --raw-dir "$(RAW_DIR)"

download-sgd:
	$(ENV) $(PYTHON) -m agentic_eval_framework.data.download_sgd --out-dir "$(RAW_DIR)"

build-data:
	$(ENV) $(PYTHON) scripts/01_build_dataset.py --use-sample

build-data-real: validate-raw
	$(ENV) $(PYTHON) scripts/01_build_dataset.py --raw-dir "$(RAW_DIR)" $(MAX_DIALOGUES_ARG)

train:
	$(ENV) $(PYTHON) scripts/02_train_model.py --backend sklearn

train-gpu:
	$(ENV) $(PYTHON) scripts/02_train_model.py --backend transformer --model-name "$(MODEL_NAME)"

evaluate:
	$(ENV) $(PYTHON) scripts/03_run_evaluation.py

replay-failures:
	$(ENV) $(PYTHON) scripts/04_replay_failures.py

export-reports:
	$(ENV) $(PYTHON) scripts/05_export_reports.py

test:
	$(ENV) pytest -q

smoke-pipeline:
	$(ENV) $(PYTHON) scripts/run_smoke_pipeline.py

smoke: smoke-pipeline test

audit:
	$(ENV) $(PYTHON) scripts/06_project_audit.py

smoke-full: smoke audit

real-data-pipeline:
	$(ENV) $(PYTHON) scripts/11_run_real_data_pipeline.py --raw-dir "$(RAW_DIR)" --backend "$(BACKEND)" --model-name "$(MODEL_NAME)" $(MAX_DIALOGUES_ARG)

clean-artifacts:
	$(ENV) $(PYTHON) scripts/07_clean_artifacts.py

package-release:
	$(ENV) $(PYTHON) scripts/08_package_release.py

monte-carlo:
	$(ENV) $(PYTHON) scripts/09_run_monte_carlo.py

monte-carlo-small:
	$(ENV) $(PYTHON) scripts/09_run_monte_carlo.py --train-dialogues 300 --dev-dialogues 120 --replications 4 --scenarios-per-replication 40 --concurrency 16

monte-carlo-gpu:
	$(ENV) $(PYTHON) scripts/09_run_monte_carlo.py --backend transformer --model-name distilbert-base-uncased --epochs 2 --batch-size 16

verify-package:
	$(ENV) $(PYTHON) scripts/10_verify_package.py

api:
	$(ENV) uvicorn agentic_eval_framework.api.app:app --host 0.0.0.0 --port 8000

release-check: smoke-full verify-package public-audit

public-audit:
	$(ENV) $(PYTHON) scripts/20_public_release_audit.py
