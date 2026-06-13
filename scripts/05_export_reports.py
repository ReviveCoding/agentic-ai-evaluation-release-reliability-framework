from __future__ import annotations

# Allow running scripts directly without requiring `pip install -e .` first.
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from agentic_eval_framework.reporting.export_reports import export_all_reports


if __name__ == "__main__":
    export_all_reports()
    print("Exported reports.")
