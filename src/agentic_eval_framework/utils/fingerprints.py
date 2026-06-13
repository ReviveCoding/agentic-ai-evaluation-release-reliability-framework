from __future__ import annotations

import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_paths(paths: Iterable[str | Path]) -> str:
    digest = hashlib.sha256()
    normalized = sorted(Path(p) for p in paths)
    for path in normalized:
        if not path.exists() or not path.is_file():
            continue
        digest.update(str(path.name).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def stable_json_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def package_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    versions["python"] = platform.python_version()
    return versions
