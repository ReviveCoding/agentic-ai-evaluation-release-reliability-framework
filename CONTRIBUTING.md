# Contributing

## Development setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
```

## Before opening a pull request

```bash
python -m pytest -q
python scripts/06_project_audit.py
python scripts/10_verify_package.py
python scripts/20_public_release_audit.py
```

Keep changes focused. Add tests for behavior changes. Do not commit raw
downloaded datasets, trained weights, local traces, credentials, or generated
output directories.

Changes to evaluation metrics, release gates, replay signatures, or safety
logic must include regression coverage and documentation updates.

Contributions are licensed under the Apache License 2.0.
