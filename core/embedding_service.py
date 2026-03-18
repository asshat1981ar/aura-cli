"""EmbeddingService — extracted from ModelAdapter (B1).

Owns all embedding configuration, provider dispatch, rate limiting,
and the embed()/get_embedding()/model_id()/dimensions()/healthcheck() interface.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from typing import Any, Dict, List, TYPE_CHECKING

from core.config_manager import config
from core.file_tools import _aura_safe_loads
from core.logging_utils import log_json
from core.runtime_auth import resolve_local_model_profiles, resolve_openai_api_key
from memory.embedding_provider import LocalEmbeddingProvider

if TYPE_CHECKING:
    import numpy as np


def _np():
    """Lazy import of numpy — avoids 2-4s startup penalty on Termux."""
    import numpy as _numpy
    return _numpy


# ---------------------------------------------------------------------------
# Rate limiter (thread-safe token bucket)
# ---------------------------------------------------------------------------

class _TokenBucketLimiter:
    """Thread-safe token bucket for remote embedding calls."""

    def __init__(self, *, tokens_per_second: float, capacity: float) -> None:
        self.tokens_per_second = tokens_per_second
        self.capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self._event = threading.Event()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                if elapsed > 0:
                    self._tokens = min(self.capacity, self._tokens + elapsed * self.tokens_per_second)
                    self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_time = max((1.0 - self._tokens) / self.tokens_per_second, 0.0)
            self._event.wait(timeout=wait_time)
            self._event.clear()


_EMBEDDING_LIMITER = _TokenBucketLimiter(tokens_per_second=4.0, capacity=1.0)


# ---------------------------------------------------------------------------
# EmbeddingService
# ---------------------------------------------------------------------------

class EmbeddingService:
    """Manages embedding generation across OpenAI, local profiles, and local TF-IDF/SVD."""

    def __init__(self, *, make_request_with_retries=None) -> None:
        """
        Args:
            make_request_with_retries: callable matching ModelAdapter._make_request_with_retries
                                       signature.  If None, uses requests directly.
        """
        self._make_request_with_retries = make_request_with_retries
        self._embedding_disabled = False
        self._embedding_disabled_reason: str | None = None
        self._embedding_disabled_logged = False
        self._local_embedding_provider = LocalEmbeddingProvider()

        # Configuration
        semantic_memory = config.get("semantic_memory", {}) or {}
        configured_embedding_model = (
            semantic_memory.get("embedding_model")
            or config.get("model_routing", {}).get("embedding")
            or "text-embedding-3-small"
        )
        if isinstance(configured_embedding_model, str) and configured_embedding_model.startswith("openai/"):
            configured_embedding_model = configured_embedding_model.split("/", 1)[1]
        self._embedding_model: str = configured_embedding_model
        self._embedding_dims: int = 1536
        self._embedding_profile_name: str | None = None
        self._embedding_mode: str = "openai"
        self._configure_embedding_backend()

    # ------------------------------------------------------------------
    # Profile helpers (thin wrappers around runtime_auth)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_local_profiles() -> dict[str, dict]:
        return resolve_local_model_profiles()

    def _resolve_local_profile_name(self, route_key: str) -> str | None:
        raw = config.get("local_model_routing", {}) or {}
        if not isinstance(raw, dict):
            return None
        profile_name = raw.get(route_key)
        if isinstance(profile_name, str) and profile_name in self._get_local_profiles():
            return profile_name
        return None

    @staticmethod
    def _profile_timeout(profile: dict, *, key: str, default: float) -> float:
        try:
            value = float(profile.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    @staticmethod
    def _profile_retries(profile: dict) -> int:
        try:
            value = int(profile.get("retries", 3))
        except (TypeError, ValueError):
            return 3
        return value if value > 0 else 3

    @staticmethod
    def _profile_backoff(profile: dict) -> float:
        try:
            value = float(profile.get("backoff_factor", 0.5))
        except (TypeError, ValueError):
            return 0.5
        return value if value >= 0 else 0.5

    # ------------------------------------------------------------------
    # Backend configuration
    # ------------------------------------------------------------------

    def _configure_embedding_backend(self) -> None:
        semantic_memory = config.get("semantic_memory", {}) or {}
        embedding_model = semantic_memory.get("embedding_model")

        if isinstance(embedding_model, str) and embedding_model.startswith("local_profile:"):
            profile_name = embedding_model.split(":", 1)[1]
            if profile_name in self._get_local_profiles():
                self._embedding_profile_name = profile_name
                self._embedding_mode = "local_profile"
        elif embedding_model in {"local-tfidf-svd-50d", "local/tfidf-svd-50d"}:
            self._embedding_model = "local-tfidf-svd-50d"
            self._embedding_dims = self._local_embedding_provider.dimensions()
            self._embedding_mode = "local_builtin"
        else:
            profile_name = self._resolve_local_profile_name("embedding")
            if profile_name is not None:
                self._embedding_profile_name = profile_name
                self._embedding_mode = "local_profile"

        if self._embedding_profile_name:
            profile = self._get_local_profiles().get(self._embedding_profile_name, {})
            profile_model = profile.get("embedding_model") or profile.get("model")
            if isinstance(profile_model, str) and profile_model.strip():
                self._embedding_model = profile_model
            dims = profile.get("embedding_dims")
            if dims is not None:
                try:
                    self._embedding_dims = int(dims)
                except (TypeError, ValueError):
                    pass

    # ------------------------------------------------------------------
    # Local profile embedding dispatchers
    # ------------------------------------------------------------------

    def _call_local_openai_embeddings(self, profile: dict, texts: List[str]) -> List[np.ndarray]:
        base_url = str(profile.get("base_url") or "http://127.0.0.1:8080/v1").rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        url = f"{base_url}/embeddings"
        model = profile.get("embedding_model") or profile.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Local openai_compatible embedding profile requires `embedding_model` or `model`.")

        headers = {"Content-Type": "application/json"}
        api_key = profile.get("api_key")
        if isinstance(api_key, str) and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"

        payload: dict[str, Any] = {"model": model, "input": texts}
        response = self._do_request(
            "POST", url, headers, payload,
            retries=self._profile_retries(profile),
            backoff_factor=self._profile_backoff(profile),
            timeout=self._profile_timeout(profile, key="request_timeout_seconds", default=60.0),
            retry_label=f"local_openai_embeddings:{model}",
        )
        data = response.json()
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        vectors = [_np().array(item["embedding"], dtype="float32") for item in sorted_data]
        if vectors:
            self._embedding_dims = int(vectors[0].shape[0])
        return vectors

    def _call_local_command_embeddings(self, profile: dict, texts: List[str]) -> List[np.ndarray]:
        command = profile.get("embedding_command") or profile.get("command")
        if isinstance(command, str):
            command_parts = shlex.split(command)
        elif isinstance(command, list) and all(isinstance(part, str) for part in command):
            command_parts = list(command)
        else:
            raise ValueError("Local command embedding profile requires `embedding_command` or `command`.")

        input_json = json.dumps({"texts": texts})
        use_stdin = True
        rendered_parts: list[str] = []
        for part in command_parts:
            if "{input_json}" in part:
                use_stdin = False
                rendered_parts.append(part.replace("{input_json}", input_json))
            else:
                rendered_parts.append(part)

        result = subprocess.run(
            rendered_parts,
            input=input_json if use_stdin else None,
            capture_output=True,
            text=True,
            check=True,
            timeout=self._profile_timeout(profile, key="subprocess_timeout_seconds", default=120.0),
        )
        payload = _aura_safe_loads(result.stdout, "local_embedding_command")
        if isinstance(payload, dict) and "data" in payload:
            payload = [item.get("embedding") for item in payload["data"]]
        if not isinstance(payload, list):
            raise ValueError("Embedding command must return a JSON array or OpenAI-style data object.")
        vectors = [_np().array(item, dtype="float32") for item in payload]
        if vectors:
            self._embedding_dims = int(vectors[0].shape[0])
        return vectors

    def _call_local_ollama_embeddings(self, profile: dict, texts: List[str]) -> List[np.ndarray]:
        base_url = str(profile.get("base_url") or "http://127.0.0.1:11434").rstrip("/")
        model = profile.get("embedding_model") or profile.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Local ollama embedding profile requires `embedding_model` or `model`.")

        headers = {"Content-Type": "application/json"}
        payload = {"model": model, "input": texts}
        try:
            response = self._do_request(
                "POST", f"{base_url}/api/embed", headers, payload,
                retries=self._profile_retries(profile),
                backoff_factor=self._profile_backoff(profile),
                timeout=self._profile_timeout(profile, key="request_timeout_seconds", default=60.0),
                retry_label=f"local_ollama_embeddings:{model}",
            )
            data = response.json()
            embeddings = data.get("embeddings")
            if isinstance(embeddings, list):
                vectors = [_np().array(item, dtype="float32") for item in embeddings]
                if vectors:
                    self._embedding_dims = int(vectors[0].shape[0])
                return vectors
        except Exception:
            pass

        vectors: list[np.ndarray] = []
        for text in texts:
            response = self._do_request(
                "POST", f"{base_url}/api/embeddings", headers,
                {"model": model, "prompt": text},
                retries=self._profile_retries(profile),
                backoff_factor=self._profile_backoff(profile),
                timeout=self._profile_timeout(profile, key="request_timeout_seconds", default=60.0),
                retry_label=f"local_ollama_embeddings:{model}",
            )
            data = response.json()
            vectors.append(_np().array(data["embedding"], dtype="float32"))
        if vectors:
            self._embedding_dims = int(vectors[0].shape[0])
        return vectors

    def _embed_with_local_profile(self, texts: List[str]) -> List[np.ndarray]:
        if not self._embedding_profile_name:
            raise ValueError("No local embedding profile configured.")
        profile = self._get_local_profiles().get(self._embedding_profile_name)
        if not isinstance(profile, dict):
            raise ValueError(f"Unknown local embedding profile: {self._embedding_profile_name}")

        provider = str(profile.get("provider") or "openai_compatible")
        if provider == "openai_compatible":
            return self._call_local_openai_embeddings(profile, texts)
        if provider == "command":
            return self._call_local_command_embeddings(profile, texts)
        if provider == "ollama":
            return self._call_local_ollama_embeddings(profile, texts)
        raise ValueError(f"Unsupported local embedding provider: {provider}")

    # ------------------------------------------------------------------
    # Request helper
    # ------------------------------------------------------------------

    def _do_request(self, method, url, headers, json_payload, **kwargs):
        """Delegates to the injected retry helper or falls back to plain requests."""
        if self._make_request_with_retries:
            return self._make_request_with_retries(method, url, headers, json_payload, **kwargs)
        import requests
        response = requests.request(method, url, headers=headers, json=json_payload,
                                    timeout=kwargs.get("timeout", 60))
        response.raise_for_status()
        return response

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def model_id(self) -> str:
        return self._embedding_model

    def dimensions(self) -> int:
        return self._embedding_dims

    def healthcheck(self) -> bool:
        try:
            self.embed(["test"])
            return True
        except Exception:
            return False

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Generate vector embeddings for a list of texts."""
        if not texts:
            return []

        if self._embedding_mode == "local_builtin":
            return [_np().array(vec, dtype="float32") for vec in self._local_embedding_provider.embed(texts)]

        if self._embedding_mode == "local_profile":
            try:
                return self._embed_with_local_profile(texts)
            except Exception as exc:
                log_json(
                    "WARN",
                    "search_embedding_local_profile_failed",
                    details={"error": str(exc), "profile": self._embedding_profile_name, "fallback": "local_tfidf_svd"},
                )
                self._embedding_model = "local-tfidf-svd-50d"
                self._embedding_dims = self._local_embedding_provider.dimensions()
                self._embedding_mode = "local_builtin"
                return [_np().array(vec, dtype="float32") for vec in self._local_embedding_provider.embed(texts)]

        openai_api_key = resolve_openai_api_key() or os.environ.get("OPENAI_API_KEY")
        if not openai_api_key or self._embedding_disabled:
            reason = self._embedding_disabled_reason or "OPENAI_API_KEY not set for OpenAI embedding call."
            if not self._embedding_disabled_logged:
                log_json("WARN", "search_embedding_failed", details={"error": reason})
                self._embedding_disabled_logged = True
            return [_np().zeros(self._embedding_dims, dtype="float32") for _ in texts]

        _EMBEDDING_LIMITER.acquire()

        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {"input": texts, "model": self._embedding_model}

        try:
            response = self._do_request("POST", url, headers, payload)
            data = response.json()
        except Exception as exc:
            self._embedding_disabled = True
            self._embedding_disabled_reason = f"embedding provider disabled after failure: {exc}"
            log_json(
                "WARN",
                "search_embedding_provider_disabled",
                details={"error": str(exc), "fallback": "zero_vectors"},
            )
            self._embedding_disabled_logged = True
            return [_np().zeros(self._embedding_dims, dtype="float32") for _ in texts]

        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [_np().array(item["embedding"], dtype="float32") for item in sorted_data]

    def get_embedding(self, text: str) -> np.ndarray:
        """Legacy wrapper around embed."""
        return self.embed([text])[0]
