from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def _extract_version(stdout: str) -> str | None:
    try:
        payload = json.loads(stdout.strip().splitlines()[-1])
        return str(payload.get("version"))
    except Exception:
        return None


def verify_package(root: Path = ROOT_DIR) -> dict:
    with tempfile.TemporaryDirectory(prefix="agentic_pkg_verify_") as tmp:
        tmp_path = Path(tmp)
        dist_dir = tmp_path / "dist"
        target_dir = tmp_path / "site"
        dist_dir.mkdir()
        wheel_cmd = [
            sys.executable, "-m", "pip", "wheel", str(root), "--no-deps",
            "--no-build-isolation", "-w", str(dist_dir),
        ]
        wheel_proc = subprocess.run(wheel_cmd, text=True, capture_output=True)
        wheels = sorted(dist_dir.glob("*.whl"))
        if wheel_proc.returncode != 0 or not wheels:
            return {
                "status": "FAIL",
                "wheel_built": False,
                "wheel_stdout": wheel_proc.stdout[-4000:],
                "wheel_stderr": wheel_proc.stderr[-4000:],
            }

        install_cmd = [
            sys.executable, "-m", "pip", "install", "--no-deps", "--target",
            str(target_dir), str(wheels[-1]),
        ]
        install_proc = subprocess.run(install_cmd, text=True, capture_output=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(target_dir)
        import_cmd = [
            sys.executable,
            "-c",
            "import json, agentic_eval_framework; "
            "from agentic_eval_framework.api.app import app; "
            "from agentic_eval_framework.simulation.monte_carlo import MonteCarloConfig; "
            "from agentic_eval_framework.models.model_manifest import load_model_manifest; "
            "from agentic_eval_framework.evaluators.retrieval_quality import retrieval_metrics; "
            "print(json.dumps({\"title\": app.title, \"config\": MonteCarloConfig.__name__, "
            "\"version\": agentic_eval_framework.__version__, "
            "\"manifest_loader\": callable(load_model_manifest), "
            "\"retrieval_metrics\": callable(retrieval_metrics)}))",
        ]
        import_proc = subprocess.run(import_cmd, text=True, capture_output=True, cwd=tmp_path, env=env)
        expected_version = None
        try:
            import tomllib
            expected_version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
        except Exception:
            pass
        verified_version = _extract_version(import_proc.stdout)
        passed = (
            install_proc.returncode == 0
            and import_proc.returncode == 0
            and expected_version is not None
            and verified_version == expected_version
        )
        return {
            "status": "PASS" if passed else "FAIL",
            "wheel_built": True,
            "wheel_name": wheels[-1].name,
            "wheel_size_bytes": wheels[-1].stat().st_size,
            "expected_version": expected_version,
            "install_returncode": install_proc.returncode,
            "import_returncode": import_proc.returncode,
            "import_stdout": import_proc.stdout.strip(),
            "verified_version": _extract_version(import_proc.stdout),
            "install_stderr": install_proc.stderr[-2000:],
            "import_stderr": import_proc.stderr[-2000:],
        }


def export(result: dict) -> None:
    reports = ROOT_DIR / "reports"
    outputs = ROOT_DIR / "outputs"
    reports.mkdir(exist_ok=True)
    outputs.mkdir(exist_ok=True)
    (outputs / "package_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Package Verification Report",
        "",
        f"Status: **{result['status']}**",
        "",
        f"- Wheel built: `{result.get('wheel_built')}`",
        f"- Wheel: `{result.get('wheel_name', 'n/a')}`",
        f"- Wheel size bytes: `{result.get('wheel_size_bytes', 'n/a')}`",
        f"- Expected version: `{result.get('expected_version', 'n/a')}`",
        f"- Verified installed version: `{result.get('verified_version', 'n/a')}`",
        f"- Installed-wheel import return code: `{result.get('import_returncode', 'n/a')}`",
        f"- Import output: `{result.get('import_stdout', '')}`",
    ]
    (reports / "package_verification_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = verify_package()
    export(result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
