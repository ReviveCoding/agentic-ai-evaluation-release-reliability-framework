from pathlib import Path


def test_scripts_have_direct_import_bootstrap():
    for script in Path("scripts").glob("0*.py"):
        text = script.read_text(encoding="utf-8")
        assert "sys.path.insert(0, str(SRC_DIR))" in text
