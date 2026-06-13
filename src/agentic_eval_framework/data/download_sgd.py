from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


SGD_ZIP_URL = "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/archive/refs/heads/master.zip"
EXPECTED_SPLITS = ("train", "dev", "test")


def normalize_extracted_sgd(extract_root: str | Path, target_raw_dir: str | Path = "data/raw/sgd") -> Path:
    """Normalize the downloaded SGD repository into the framework raw layout.

    Expected output:
        target_raw_dir/train/schema.json
        target_raw_dir/train/dialogues_*.json
        target_raw_dir/dev/schema.json
        target_raw_dir/test/schema.json, when available

    The public repository can be nested after zip extraction, so this function
    searches for split folders rather than assuming a fixed top-level path.
    """
    root = Path(extract_root)
    target = Path(target_raw_dir)
    target.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for split in EXPECTED_SPLITS:
        candidates = sorted(root.glob(f"**/{split}/schema.json"))
        if not candidates:
            continue
        split_src = candidates[0].parent
        split_dst = target / split
        if split_dst.exists():
            shutil.rmtree(split_dst)
        shutil.copytree(split_src, split_dst)
        copied.append(split)
    if not copied:
        raise FileNotFoundError(f"Could not find SGD split folders under {root}")
    return target


def download_sgd(out_dir: str | Path = "data/raw/sgd", cache_dir: str | Path = "data/raw/sgd_download") -> Path:
    """Download and normalize the public SGD repository.

    The default pipeline does not require network access because sample SGD-style
    data is included. Use this for full public-data experiments locally.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / "sgd_master.zip"
    print(f"Downloading {SGD_ZIP_URL} -> {zip_path}")
    urllib.request.urlretrieve(SGD_ZIP_URL, zip_path)
    extract_dir = cache / "extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    normalized = normalize_extracted_sgd(extract_dir, out_dir)
    print(f"Normalized SGD data to {normalized}")
    return normalized


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/raw/sgd")
    parser.add_argument("--cache-dir", default="data/raw/sgd_download")
    args = parser.parse_args()
    download_sgd(args.out_dir, args.cache_dir)
