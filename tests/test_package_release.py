import importlib.util
import zipfile
from pathlib import Path


def load_package_release():
    script = Path("scripts/08_package_release.py")
    spec = importlib.util.spec_from_file_location("package_release_script", script)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.package_release


def test_package_release_excludes_generated_artifacts(tmp_path):
    package_release = load_package_release()
    out = package_release(tmp_path / "release.zip")
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    assert not any("__pycache__" in n for n in names)
    assert not any(".pytest_cache" in n for n in names)
    assert not any(n.endswith("traces.sqlite") for n in names)
    assert not any("/build/" in n or n.endswith("/build") for n in names)
    assert not any(".egg-info/" in n for n in names)
    assert not any("/dist/" in n for n in names)


def test_package_release_keeps_validation_reports_but_excludes_generated_mc_data(tmp_path):
    package_release = load_package_release()
    out = package_release(tmp_path / "release.zip")
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    assert any(n.endswith("reports/monte_carlo_validation_report.md") for n in names)
    assert any(n.endswith("reports/monte_carlo_summary.json") for n in names)
    assert not any("data/monte_carlo_raw" in n for n in names)
    assert not any("data/monte_carlo_processed" in n for n in names)


def test_package_release_uses_stable_archive_root(tmp_path):
    package_release = load_package_release()
    out = package_release(tmp_path / "release.zip")
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    assert names
    assert all(name.startswith("agentic-eval-framework/") for name in names)
