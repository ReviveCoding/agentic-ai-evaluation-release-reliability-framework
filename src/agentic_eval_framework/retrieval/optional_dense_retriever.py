"""Optional dense retrieval hook.

The default framework uses BM25 to keep the project lean and reproducible. This
module is intentionally a stub extension point. A local implementation can use
sentence-transformers embeddings when installed.
"""

from __future__ import annotations


def is_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False
