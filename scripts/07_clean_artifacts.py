from __future__ import annotations

from pathlib import Path
import shutil
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CLEAN_DIRS = [
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
]
GENERATED_PATHS = [
    "outputs",
    "models/tool_policy",
    "data/monte_carlo_processed",
    "data/monte_carlo_raw",
    "models/monte_carlo_tool_policy",
]


def clean(root: Path = ROOT_DIR, include_reports: bool = False) -> list[str]:
    removed: list[str] = []
    for path in root.rglob("*"):
        if path.is_dir() and (path.name in CLEAN_DIRS or path.name.endswith(".egg-info")):
            shutil.rmtree(path, ignore_errors=True)
            removed.append(str(path.relative_to(root)))
    for rel in GENERATED_PATHS:
        path = root / rel
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink()
            removed.append(rel)
    if include_reports:
        reports = root / "reports"
        if reports.exists():
            for item in reports.glob("*.md"):
                item.unlink()
                removed.append(str(item.relative_to(root)))
    return sorted(set(removed))


if __name__ == "__main__":
    for item in clean():
        print(f"removed {item}")
