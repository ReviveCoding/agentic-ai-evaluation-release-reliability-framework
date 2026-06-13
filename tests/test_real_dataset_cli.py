from pathlib import Path
import subprocess
import sys

from agentic_eval_framework.data.parse_sgd import create_sample_raw

ROOT = Path(__file__).resolve().parents[1]


def test_raw_dataset_validator_cli_accepts_external_dataset(tmp_path: Path) -> None:
    raw = tmp_path / "mounted_sgd"
    report = tmp_path / "report.json"
    create_sample_raw(raw)
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "00_validate_raw_dataset.py"),
            "--raw-dir",
            str(raw),
            "--report",
            str(report),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert report.exists()
    assert '"status": "PASS"' in report.read_text(encoding="utf-8")
