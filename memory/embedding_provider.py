"""Local fallback embedding provider used by ModelAdapter.

This keeps import-time dependencies light and provides a deterministic
non-network embedding backend for tests and offline workflows.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, List


class LocalEmbeddingProvider:
    """Generate deterministic fixed-size local embeddings.

    The vectors are intentionally simple: they are stable across runs,
    require no external model downloads, and satisfy the interface expected by
    ``core.model_adapter.ModelAdapter``.
    """

    def __init__(self, dims: int = 50):
        self._dims = dims

    def dimensions(self) -> int:
        return self._dims

    def embed(self, texts: Iterable[str]) -> List[list[float]]:
        vectors: List[list[float]] = []
        for text in texts:
            digest = hashlib.sha256((text or "").encode("utf-8")).digest()
            values = [((digest[i % len(digest)] / 255.0) * 2.0) - 1.0 for i in range(self._dims)]
            vectors.append(values)
        return vectors
