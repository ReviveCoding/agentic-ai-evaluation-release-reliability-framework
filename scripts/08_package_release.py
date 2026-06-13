from __future__ import annotations

from pathlib import Path
import shutil
import sys
import zipfile

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

EXCLUDE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".git", ".mypy_cache", ".ruff_cache", ".venv", "venv", "build", "dist"}
EXCLUDE_TOP_LEVEL = {"outputs", "models"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".sqlite"}
EXCLUDE_REL_PREFIXES = {("data", "monte_carlo_raw"), ("data", "monte_carlo_processed")}
ARCHIVE_ROOT = Path("agentic-eval-framework")


def should_include(path: Path) -> bool:
    rel = path.relative_to(ROOT_DIR)
    parts = set(rel.parts)
    if parts & EXCLUDE_DIR_NAMES:
        return False
    if any(part.endswith(".egg-info") for part in rel.parts):
        return False
    if rel.parts and rel.parts[0] in EXCLUDE_TOP_LEVEL:
        return False
    if any(rel.parts[: len(prefix)] == prefix for prefix in EXCLUDE_REL_PREFIXES):
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    return True


def package_release(out_path: str | Path = "/mnt/data/agentic-eval-framework-final.zip") -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in ROOT_DIR.rglob("*"):
            if path.is_file() and should_include(path):
                zf.write(path, ARCHIVE_ROOT / path.relative_to(ROOT_DIR))
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/data/agentic-eval-framework-final.zip")
    print(package_release(target))
