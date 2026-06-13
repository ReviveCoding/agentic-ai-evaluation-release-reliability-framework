from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agentic_eval_framework.utils.io import ensure_dir

REQUIRED_PATHS = [
    "README.md", "pyproject.toml", "requirements.txt", "requirements-dev.txt",
    "requirements-gpu.txt", "requirements-dense.txt", "Makefile", "Dockerfile", ".dockerignore", ".gitignore",
    ".github/workflows/ci.yml", ".github/workflows/real-data.yml", "configs/dataset.yaml", "configs/model_registry.yaml",
    "configs/tool_registry.yaml", "configs/evaluator_registry.yaml", "configs/release_gates.yaml",
    "src/agentic_eval_framework/data/dataset_contract.py",
    "src/agentic_eval_framework/data/build_tool_policy_dataset.py",
    "src/agentic_eval_framework/models/train_tool_policy.py",
    "src/agentic_eval_framework/models/calibration.py",
    "src/agentic_eval_framework/models/model_manifest.py",
    "src/agentic_eval_framework/data/validate_dataset.py",
    "src/agentic_eval_framework/evaluators/retrieval_quality.py",
    "src/agentic_eval_framework/engine/execution_engine.py",
    "src/agentic_eval_framework/engine/replay.py",
    "src/agentic_eval_framework/engine/release_gate.py",
    "src/agentic_eval_framework/storage/sqlite_store.py",
    "src/agentic_eval_framework/simulation/monte_carlo.py",
    "src/agentic_eval_framework/api/app.py",
    "scripts/00_validate_raw_dataset.py", "scripts/01_build_dataset.py", "scripts/02_train_model.py", "scripts/03_run_evaluation.py",
    "scripts/04_replay_failures.py", "scripts/05_export_reports.py", "scripts/06_project_audit.py",
    "scripts/07_clean_artifacts.py", "scripts/08_package_release.py", "scripts/09_run_monte_carlo.py",
    "scripts/10_verify_package.py", "scripts/11_run_real_data_pipeline.py", "scripts/run_smoke_pipeline.py",
    "tests/test_execution_engine.py", "tests/test_monte_carlo.py", "tests/test_fault_injection.py",
    "tests/test_replay_reexecution.py", "tests/test_api.py", "tests/test_aggregate_weighting.py",
    "tests/test_retrieval_integration.py", "tests/test_model_manifest.py",
    "tests/test_dataset_validation.py", "tests/test_monte_carlo_edge_cases.py",
    "tests/test_release_gate_validation.py",
    "tests/test_three_way_split.py", "tests/test_fault_coupling.py",
    "tests/test_missing_split_fails_fast.py",
    "tests/test_dataset_contract.py", "tests/test_real_dataset_cli.py",
    "src/agentic_eval_framework/retrieval/recency_query.py",
    "src/agentic_eval_framework/retrieval/top1_reranker.py",
    "scripts/17_train_retrieval_reranker.py",
    "scripts/18_compare_top1_hardening.py",
    "scripts/19_evaluate_retrieval_top1.py",
    "tests/test_recency_tool_reranking.py",
    "tests/test_lightweight_reranker.py",
    "src/agentic_eval_framework/retrieval/hierarchical_retriever.py",
    "src/agentic_eval_framework/models/ablation.py",
    "scripts/15_run_ablation.py",
    "scripts/16_compare_hardening.py",
    "tests/test_hierarchical_retrieval.py",
    "tests/test_replay_stability.py",
    "tests/test_ablation_inputs.py",
    "src/agentic_eval_framework/data/sgd_stratified.py",
    "scripts/12_build_stratified_sgd.py",
    "scripts/13_evaluate_ood.py",
    "scripts/14_run_stratified_real_data_pipeline.py",
    "tests/test_stratified_sgd.py",
]

EXPECTED_REPORTS = [
        "reports/ablation_report.md",
        "reports/dataset_card.md",
        "reports/final_release_manifest.json",
        "reports/final_release_summary.md",
        "reports/final_retrieval_top1_hardening.md",
        "reports/monte_carlo_summary.json",
        "reports/monte_carlo_validation_report.md",
        "reports/package_verification_report.md",
        "reports/public_release_audit.json",
        "reports/public_release_audit.md",
    ]


def audit_project(root: Path = ROOT_DIR) -> dict:
    missing = [p for p in REQUIRED_PATHS if not (root / p).exists()]
    missing_reports = [p for p in EXPECTED_REPORTS if not (root / p).exists()]
    cache_dirs = sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_dir()
        and p.name in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
        and not any(part in {".venv", "build", "dist", "_patch_backups"} for part in p.relative_to(root).parts)
    )
    warnings: list[str] = []
    if cache_dirs:
        warnings.append("Development cache directories are present in the working tree; release packaging excludes them.")
    version_consistent = False
    try:
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib
        pyproject_version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
        from agentic_eval_framework import __version__
        version_consistent = pyproject_version == __version__
        if not version_consistent:
            warnings.append(f"Version mismatch: pyproject={pyproject_version}, package={__version__}")
    except Exception as exc:
        warnings.append(f"Could not validate package version consistency: {exc}")

    retrieval_integration_present = (root / "src/agentic_eval_framework/evaluators/retrieval_quality.py").exists()
    ci_text = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8") if (root / ".github/workflows/ci.yml").exists() else ""
    ci_monte_carlo_present = "make monte-carlo-small" in ci_text
    real_data_ci_present = (root / ".github/workflows/real-data.yml").exists()

    result = {
        "missing_required_paths": missing,
        "missing_expected_reports": missing_reports,
        "runnable_commands": [
            "python scripts/00_validate_raw_dataset.py --raw-dir /path/to/sgd",
            "python scripts/11_run_real_data_pipeline.py --raw-dir /path/to/sgd --backend sklearn",
            "python scripts/14_run_stratified_real_data_pipeline.py --raw-dir /path/to/sgd --backend sklearn --max-dialogues 250",
            "python scripts/01_build_dataset.py --use-sample",
            "python scripts/02_train_model.py --backend sklearn",
            "python scripts/03_run_evaluation.py",
            "python scripts/04_replay_failures.py",
            "python scripts/05_export_reports.py",
            "python scripts/09_run_monte_carlo.py --replications 4 --scenarios-per-replication 40",
            "python scripts/10_verify_package.py",
            "pytest -q",
            "make release-check",
        ],
        "github_actions_present": (root / ".github/workflows/ci.yml").exists(),
        "dockerfile_present": (root / "Dockerfile").exists(),
        "dockerignore_present": (root / ".dockerignore").exists(),
        "gpu_training_path_present": (root / "requirements-gpu.txt").exists() and (root / "scripts/run_gpu_pipeline.ps1").exists(),
        "cache_dirs_present": bool(cache_dirs),
        "version_consistent": version_consistent,
        "retrieval_integration_present": retrieval_integration_present,
        "ci_monte_carlo_present": ci_monte_carlo_present,
        "real_data_ci_present": real_data_ci_present,
        "cache_dirs": cache_dirs,
        "warnings": warnings,
        "status": "PASS" if not missing and not missing_reports and version_consistent and retrieval_integration_present and ci_monte_carlo_present and real_data_ci_present else "REVIEW",
    }
    return result


def export_audit_report(result: dict, out_path: Path = ROOT_DIR / "reports/project_audit_report.md") -> None:
    ensure_dir(out_path.parent)
    lines = ["# Project Audit Report", "", f"Status: **{result['status']}**", ""]
    lines.append("## Required path check")
    lines.extend(f"- Missing: `{p}`" for p in result["missing_required_paths"]) if result["missing_required_paths"] else lines.append("- All required source/config/test/devops paths are present.")
    lines += ["", "## Expected reports"]
    lines.extend(f"- Missing report: `{p}`" for p in result["missing_expected_reports"]) if result["missing_expected_reports"] else lines.append("- All expected reports are present.")
    lines += ["", "## Local/GitHub runnable commands"]
    lines.extend(f"- `{cmd}`" for cmd in result["runnable_commands"])
    lines += ["", "## Deployment/reproducibility signals"]
    for key in ["github_actions_present", "dockerfile_present", "dockerignore_present", "gpu_training_path_present", "cache_dirs_present", "version_consistent", "retrieval_integration_present", "ci_monte_carlo_present", "real_data_ci_present"]:
        lines.append(f"- {key}: {result[key]}")
    if result["warnings"]:
        lines += ["", "## Warnings"] + [f"- {w}" for w in result["warnings"]]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = ensure_dir(ROOT_DIR / "outputs")
    (outputs / "project_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    result = audit_project()
    export_audit_report(result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
