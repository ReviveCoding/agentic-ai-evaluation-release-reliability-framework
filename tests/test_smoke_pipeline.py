from pathlib import Path


def test_smoke_pipeline_script_exists():
    assert Path("scripts/run_smoke_pipeline.py").exists()
