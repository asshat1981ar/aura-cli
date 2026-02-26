"""ASCM v2: EmbeddingProvider protocol and implementations."""
from __future__ import annotations

import math
import os
import random
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass


class EmbeddingProvider:
    """Protocol for embedding providers."""

    def embed(self, texts: List[str]) -> List[list]:
        raise NotImplementedError

    def model_id(self) -> str:
        raise NotImplementedError

    def dimensions(self) -> int:
        raise NotImplementedError

    def available(self) -> bool:
        raise NotImplementedError


def _random_unit_vector(dims: int, seed: int) -> list:
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(dims)]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class LocalEmbeddingProvider(EmbeddingProvider):
    """TF-IDF + SVD embeddings via scikit-learn, 50-dim. Falls back to random unit vectors."""

    DIMS = 50

    def __init__(self):
        self._vectorizer = None
        self._svd = None
        self._fitted = False
        self._sklearn_available = self._check_sklearn()

    def _check_sklearn(self) -> bool:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
            from sklearn.decomposition import TruncatedSVD  # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_fitted(self, texts: List[str]) -> bool:
        if not self._sklearn_available:
            return False
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            n_components = min(self.DIMS, max(1, len(texts) - 1))
            vectorizer = TfidfVectorizer(max_features=2000)
            tfidf = vectorizer.fit_transform(texts)
            if tfidf.shape[1] < 2 or tfidf.shape[0] < 2:
                return False
            n_components = min(n_components, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
            if n_components < 1:
                return False
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            svd.fit(tfidf)
            self._vectorizer = vectorizer
            self._svd = svd
            self._fitted = True
            return True
        except Exception:
            return False

    def embed(self, texts: List[str]) -> List[list]:
        if not texts:
            return []
        try:
            if self._sklearn_available:
                if not self._fitted:
                    self._ensure_fitted(texts)
                if self._fitted and self._vectorizer is not None and self._svd is not None:
                    tfidf = self._vectorizer.transform(texts)
                    reduced = self._svd.transform(tfidf)
                    result = []
                    for row in reduced:
                        vec = list(float(x) for x in row)
                        # Pad or trim to DIMS
                        if len(vec) < self.DIMS:
                            vec = vec + [0.0] * (self.DIMS - len(vec))
                        else:
                            vec = vec[: self.DIMS]
                        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                        result.append([x / norm for x in vec])
                    return result
        except Exception:
            pass
        # Fallback: random unit vectors deterministically seeded by content hash
        return [_random_unit_vector(self.DIMS, hash(t) & 0xFFFFFFFF) for t in texts]

    def model_id(self) -> str:
        return "local-tfidf-svd-50d"

    def dimensions(self) -> int:
        return self.DIMS

    def available(self) -> bool:
        return True


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Uses text-embedding-3-small via ModelAdapter. Falls back to LocalEmbeddingProvider."""

    DIMS = 1536

    def __init__(self, model_adapter=None):
        self._adapter = model_adapter
        self._fallback = LocalEmbeddingProvider()
        self._api_available = model_adapter is not None

    def embed(self, texts: List[str]) -> List[list]:
        if not texts:
            return []
        if self._api_available and self._adapter is not None:
            try:
                results = []
                for text in texts:
                    emb = self._adapter.embed(text, model="text-embedding-3-small")
                    if emb and isinstance(emb, list):
                        results.append(emb)
                    else:
                        raise ValueError("empty embedding")
                return results
            except Exception:
                pass
        return self._fallback.embed(texts)

    def model_id(self) -> str:
        return "text-embedding-3-small"

    def dimensions(self) -> int:
        if self._api_available:
            return self.DIMS
        return self._fallback.dimensions()

    def available(self) -> bool:
        return True


def get_default_provider(model_adapter=None) -> EmbeddingProvider:
    """Factory: returns OpenAIEmbeddingProvider if adapter given, else LocalEmbeddingProvider."""
    if model_adapter is not None:
        return OpenAIEmbeddingProvider(model_adapter=model_adapter)
    return LocalEmbeddingProvider()
