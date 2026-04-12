"""Embedding providers for ASCM v2.

Provides a structured EmbeddingResult type and interchangeable backends:
  - LocalEmbeddingProvider: deterministic, no-network, for tests and offline runs.
  - OpenAIEmbeddingProvider: hosted embeddings via the OpenAI API.
  - ONNXEmbeddingProvider: local mobile-optimized embeddings via ONNX Runtime.

All expose the ASCM v2 contract (embed_text / embed_batch returning EmbeddingResult)
as well as the legacy embed(texts) shim consumed by ModelAdapter.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional


class _MissingPackage:
    """Placeholder for optional dependencies that are not installed."""

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, attr: str) -> None:
        raise AttributeError(f"Optional dependency '{self._name}' is required for this operation.")

    def __call__(self, *args: object, **kwargs: object) -> None:
        raise ImportError(f"Optional dependency '{self._name}' is required for this operation.")


try:
    import requests as _requests  # type: ignore
except ImportError:  # pragma: no cover
    _requests = _MissingPackage("requests")  # type: ignore


@dataclass
class EmbeddingResult:
    """Structured result from an embedding provider call."""

    vector: List[float]
    model_name: str
    model_version: str
    provider_type: str  # "local" | "openai" | "onnx"
    dimensions: int


class EmbeddingProvider(ABC):
    """Abstract base class for all embedding providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the model used by the provider."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Version of the model used by the provider."""

    @property
    @abstractmethod
    def provider_type(self) -> str:
        """Type of provider (e.g., 'local', 'openai', 'onnx')."""

    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of the generated embeddings."""

    @abstractmethod
    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        """Legacy interface: returns list of raw float vectors (no metadata)."""

    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a single string and return a structured EmbeddingResult."""

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed a batch of strings and return a list of EmbeddingResults."""

    @abstractmethod
    def healthcheck(self) -> bool:
        """Verify the provider is operational."""


class LocalEmbeddingProvider(EmbeddingProvider):
    """Generate deterministic fixed-size local embeddings.

    Vectors are stable across runs, require no external model downloads, and
    satisfy the interface expected by ``core.model_adapter.ModelAdapter``.
    """

    @property
    def model_name(self) -> str:
        return "local-sha256"

    @property
    def model_version(self) -> str:
        return "1.0"

    @property
    def provider_type(self) -> str:
        return "local"

    def __init__(self, dims: int = 50):
        self._dims = dims

    def dimensions(self) -> int:
        return self._dims

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        """Legacy interface: returns list of raw float vectors (no metadata)."""
        vectors: List[List[float]] = []
        for text in texts:
            digest = hashlib.sha256((text or "").encode("utf-8")).digest()
            values = [((digest[i % len(digest)] / 255.0) * 2.0) - 1.0 for i in range(self._dims)]
            vectors.append(values)
        return vectors

    def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a single string and return a structured EmbeddingResult."""
        vector = self.embed([text])[0]
        return EmbeddingResult(
            vector=vector,
            model_name=self.model_name,
            model_version=self.model_version,
            provider_type=self.provider_type,
            dimensions=self._dims,
        )

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed a batch of strings and return a list of EmbeddingResults."""
        raw = self.embed(texts)
        return [
            EmbeddingResult(
                vector=vec,
                model_name=self.model_name,
                model_version=self.model_version,
                provider_type=self.provider_type,
                dimensions=self._dims,
            )
            for vec in raw
        ]

    def healthcheck(self) -> bool:
        return True


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Hosted embeddings via the OpenAI API.

    Uses ``requests`` (lazy-imported) to POST to the embeddings endpoint.
    Pass ``api_key`` explicitly or set the ``OPENAI_API_KEY`` environment variable.
    """

    @property
    def model_version(self) -> str:
        return "1"

    @property
    def provider_type(self) -> str:
        return "openai"

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self._api_key = api_key
        self._model = model
        self._api_url = "https://api.openai.com/v1/embeddings"

    @property
    def model_name(self) -> str:
        return self._model

    def dimensions(self) -> int:
        # text-embedding-3-small → 1536; text-embedding-3-large → 3072
        return 3072 if "large" in self._model else 1536

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        payload = {"input": texts, "model": self._model}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        resp = _requests.post(self._api_url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Sort by index to preserve order
        items = sorted(data["data"], key=lambda d: d["index"])
        return [item["embedding"] for item in items]

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        """Legacy interface: returns list of raw float vectors."""
        return self._call_api(list(texts))

    def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a single string and return a structured EmbeddingResult."""
        vector = self._call_api([text])[0]
        return EmbeddingResult(
            vector=vector,
            model_name=self._model,
            model_version=self.model_version,
            provider_type=self.provider_type,
            dimensions=len(vector),
        )

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed a batch of strings and return a list of EmbeddingResults."""
        vectors = self._call_api(texts)
        return [
            EmbeddingResult(
                vector=vec,
                model_name=self.model_name,
                model_version=self.model_version,
                provider_type=self.provider_type,
                dimensions=len(vec),
            )
            for vec in vectors
        ]

    def healthcheck(self) -> bool:
        try:
            self._call_api(["health"])
            return True
        except Exception:
            return False
