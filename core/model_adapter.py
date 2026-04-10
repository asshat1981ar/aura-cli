"""Unified LLM adapter with provider and caching mixins.

:class:`ModelAdapter` is the single entry-point for all LLM interactions in
AURA.  Provider-specific call methods live in :mod:`core.model_providers`
(:class:`ProvidersMixin`) and caching logic in :mod:`core.model_cache`
(:class:`CacheMixin`).  This module keeps ``__init__``, ``respond``,
``respond_for_role``, embedding helpers, and all module-level imports so that
existing ``patch("core.model_adapter.…")`` targets remain stable.
"""
from __future__ import annotations

import concurrent.futures
import os
import json
import time
from pathlib import Path
from typing import List


class _MissingPackage:
    """Placeholder for optional dependencies that are not installed."""

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, attr: str) -> None:
        raise AttributeError(
            f"Optional dependency '{self._name}' is required for this operation."
        )

    def __call__(self, *args: object, **kwargs: object) -> None:
        raise ImportError(
            f"Optional dependency '{self._name}' is required for this operation."
        )


try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover - exercised via optional-deps tests
    requests = _MissingPackage("requests")  # type: ignore

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - exercised via optional-deps tests
    np = _MissingPackage("numpy")  # type: ignore

from core.logging_utils import log_json # Import log_json
from core.file_tools import _aura_safe_loads # Import _aura_safe_loads
from core.config_manager import config
from core.runtime_auth import (
    resolve_openai_api_key,
)
from memory.embedding_provider import LocalEmbeddingProvider

# Mixin imports — keep *after* the module-level names above so that the
# mixins can reference ``core.model_adapter.requests`` etc. at call time.
from core.model_providers import ProvidersMixin
from core.model_cache import CacheMixin

# Removed dangerous global IPv4-only monkeypatch for socket.getaddrinfo.


class ModelAdapter(ProvidersMixin, CacheMixin):
    """
    The ModelAdapter manages interactions with various Large Language Models (LLMs)
    and external tools. It provides a unified interface for sending prompts,
    receiving responses, and executing tool calls. It includes a robust fallback
    mechanism and performance tracking for different LLMs.
    """

    def __init__(self):
        """
        Initializes the ModelAdapter, validates CLI paths,
        and defines an allowlist for executable tools.
        """
        # API keys will be fetched dynamically within their respective call methods
        self.gemini_cli_path = config.get("gemini_cli_path")
        self.codex_cli_path = config.get("codex_cli_path")
        self.copilot_cli_path = config.get("copilot_cli_path")
        self.mcp_server_url = config.get("mcp_server_url")
        self.router = None
        self.cache_db = None
        self.telemetry_agent = None
        self.cache_ttl = config.get("llm_timeout", 3600)
        self._embedding_disabled = False
        self._embedding_disabled_reason = None
        self._embedding_disabled_logged = False
        self._local_profile_cooldowns: dict[str, float] = {}
        self._local_profile_cooldown_reasons: dict[str, str] = {}

        # Configuration
        semantic_memory = config.get("semantic_memory", {}) or {}
        configured_embedding_model = semantic_memory.get("embedding_model") or config.get("model_routing", {}).get("embedding") or "text-embedding-3-small"
        if isinstance(configured_embedding_model, str) and configured_embedding_model.startswith("openai/"):
            configured_embedding_model = configured_embedding_model.split("/", 1)[1]
        self._embedding_model = configured_embedding_model
        self._embedding_dims = 1536
        self._local_embedding_provider = LocalEmbeddingProvider()
        self._embedding_profile_name = None
        self._embedding_mode = "openai"
        self._configure_embedding_backend()

        # Validate CLI paths
        self._validate_cli_paths()

        # In-memory cache (L0) — populated by preload_cache()
        self._mem_cache: dict = {}

        # Define an explicit allowlist for tools
        self.ALLOWED_TOOLS = {
            "search", "read_file", "list_directory", "glob",
            "get_repo", "create_issue", "get_issue_details", "update_file", "get_pull_request_details"
        }

    # ------------------------------------------------------------------
    # CLI path validation
    # ------------------------------------------------------------------

    def _validate_cli_paths(self):
        """Helper to validate external LLM CLI paths."""
        cli_configs = [
            ("gemini", self.gemini_cli_path),
            ("codex", self.codex_cli_path),
            ("copilot", self.copilot_cli_path)
        ]

        for name, path in cli_configs:
            if path:
                p = Path(path)
                if not p.is_file():
                    log_json("WARN", f"{name}_cli_not_found", details={"path": path})
                    setattr(self, f"{name}_cli_path", None)
                elif not os.access(path, os.X_OK):
                    log_json("WARN", f"{name}_cli_not_executable", details={"path": path})
                    setattr(self, f"{name}_cli_path", None)

    # ------------------------------------------------------------------
    # Embedding backend configuration
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
    # Context budget helpers
    # ------------------------------------------------------------------

    def estimate_context_budget(self, goal: str, goal_type: str = "default") -> int:
        """Returns a token budget estimate based on goal_type and goal length."""
        base_budgets = {
            "docs": 2000,
            "bug_fix": 4000,
            "feature": 6000,
            "refactor": 4000,
            "security": 5000,
            "default": 4000,
        }
        budget = base_budgets.get(goal_type, base_budgets["default"])
        extra = min(len(goal) // 10, 2000)
        return budget + extra

    def compress_context(self, text: str, max_tokens: int) -> str:
        """Truncates text to fit within max_tokens (rough estimate: 4 chars per token)."""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    # ------------------------------------------------------------------
    # Router / telemetry
    # ------------------------------------------------------------------

    def set_router(self, router):
        """Attaches an adaptive router to the adapter."""
        self.router = router
        log_json("INFO", "model_router_attached")

    def set_telemetry_agent(self, telemetry_agent):
        """Attaches a telemetry agent to the adapter."""
        self.telemetry_agent = telemetry_agent
        log_json("INFO", "telemetry_agent_attached")

    def _log_telemetry(self, model_name, latency, response_text):
        """Logs latency and estimated token count to the telemetry agent."""
        if not self.telemetry_agent:
            return
        token_count = len(response_text) // 4
        try:
            self.telemetry_agent.log(model_name, latency, token_count)
        except Exception as e:
            log_json("WARN", "telemetry_logging_failed", details={"error": str(e)})

    # ------------------------------------------------------------------
    # LLM timeout
    # ------------------------------------------------------------------

    @property
    def LLM_TIMEOUT(self) -> int:
        return config.get("llm_timeout", 60)

    def _call_with_timeout(self, fn, *args, timeout: int | None = None) -> str:
        """Run *fn(*args)* in a thread and raise TimeoutError if it exceeds *timeout* seconds."""
        _timeout = timeout if timeout is not None else self.LLM_TIMEOUT
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(fn, *args)
            try:
                return future.result(timeout=_timeout)
            except concurrent.futures.TimeoutError:
                log_json("ERROR", "llm_call_timeout",
                         details={"fn": fn.__name__, "timeout_s": _timeout})
                raise TimeoutError(
                    f"LLM call to {fn.__name__!r} exceeded {_timeout}s timeout"
                )

    # ------------------------------------------------------------------
    # respond / respond_for_role
    # ------------------------------------------------------------------

    def respond_for_role(self, route_key: str, prompt: str) -> str:
        # Try local profile first (for Android/local models)
        profile_name = self._resolve_local_profile_name(route_key)
        if profile_name:
            cache_key = f"[local-role:{route_key}|profile:{profile_name}] {prompt}"
            cached = self._get_cached_response(cache_key)
            if cached:
                return cached
            try:
                model_response = self._call_with_timeout(self.call_local_profile, profile_name, prompt)
                self._save_to_cache(cache_key, model_response)
                return model_response
            except Exception as e:
                log_json("WARN", "local_role_call_failed", details={"route_key": route_key, "profile": profile_name, "error": str(e), "fallback": "openrouter_routed"})

        # Route through OpenRouter with task-specific model selection
        routing = config.get("model_routing", {})
        if route_key in routing and config.get("primary_provider") == "openrouter":
            cache_key = f"[openrouter-role:{route_key}] {prompt}"
            cached = self._get_cached_response(cache_key)
            if cached:
                return cached
            try:
                model_response = self._call_with_timeout(self.call_openrouter, prompt, route_key)
                self._save_to_cache(cache_key, model_response)
                return model_response
            except Exception as e:
                log_json("WARN", "openrouter_role_call_failed", details={"route_key": route_key, "error": str(e), "fallback": "default_respond"})

        return self.respond(prompt)

    def respond(self, prompt: str):
        cached = self._get_cached_response(prompt)
        if cached:
            return cached
        model_response = None
        if self.router:
            try:
                model_response = self._call_with_timeout(self.router.route, prompt)
            except Exception as e:
                log_json("WARN", "router_call_failed", details={"error": str(e), "fallback": "Direct fallbacks"})
        if not model_response:
            primary = config.get("primary_provider", "openrouter")
            if primary == "openrouter":
                try:
                    model_response = self._call_with_timeout(self.call_openrouter, prompt)
                except Exception as e:
                    log_json("WARN", "openrouter_call_failed", details={"error": str(e), "fallback": "OpenAI"})
            if not model_response:
                try:
                    model_response = self._call_with_timeout(self.call_openai, prompt)
                except Exception as e:
                    log_json("WARN", "openai_call_failed", details={"error": str(e), "fallback": "Anthropic"})
                    try:
                        model_response = self._call_with_timeout(self.call_anthropic, prompt)
                    except Exception as e:
                        log_json("WARN", "anthropic_call_failed", details={"error": str(e), "fallback": "Local Model"})
                        model_response = self._call_with_timeout(self.call_local, prompt)
        if model_response is None:
            return "Error: No model successfully responded."
        self._save_to_cache(prompt, model_response)
        try:
            parsed_response = _aura_safe_loads(model_response, "model_response")
            if isinstance(parsed_response, dict) and "tool_code" in parsed_response:
                tool_call_data = parsed_response["tool_code"]
                tool_name = tool_call_data.get("name")
                args = tool_call_data.get("args", {})
                if not isinstance(tool_name, str) or not isinstance(args, dict):
                    return "Model attempted tool call with invalid structure (name or args missing/wrong type)."
                if tool_name not in self.ALLOWED_TOOLS:
                    return f"Error: Tool '{tool_name}' is not allowed by the system configuration."
                tool_output = self._execute_tool(tool_name, args)
                return f"Tool Output: {tool_output}"
            else:
                return model_response
        except json.JSONDecodeError:
            pass
        return model_response

    # ------------------------------------------------------------------
    # EmbeddingProvider Interface
    # ------------------------------------------------------------------

    def model_id(self) -> str:
        return self._embedding_model

    def dimensions(self) -> int:
        return self._embedding_dims

    def healthcheck(self) -> bool:
        """Checks if the embedding provider is reachable."""
        try:
            self.embed(["test"])
            return True
        except (ConnectionError, TimeoutError, ValueError):
            return False

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generates vector embeddings for a list of texts.
        Falls back to zero vectors when OPENAI_API_KEY is not set.
        """
        if not texts:
            return []

        if self._embedding_mode == "local_builtin":
            return [np.array(vec, dtype=np.float32) for vec in self._local_embedding_provider.embed(texts)]

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
                return [np.array(vec, dtype=np.float32) for vec in self._local_embedding_provider.embed(texts)]

        openai_api_key = resolve_openai_api_key() or os.environ.get("OPENAI_API_KEY")
        if not openai_api_key or self._embedding_disabled:
            reason = self._embedding_disabled_reason or "OPENAI_API_KEY not set for OpenAI embedding call."
            if not self._embedding_disabled_logged:
                log_json("WARN", "search_embedding_failed", details={"error": reason})
                self._embedding_disabled_logged = True
            return [np.zeros(self._embedding_dims, dtype=np.float32) for _ in texts]

        import random
        time.sleep(0.2 + random.uniform(0, 0.1))

        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "input": texts,
            "model": self._embedding_model
        }

        try:
            response = self._make_request_with_retries("POST", url, headers, payload)
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
            return [np.zeros(self._embedding_dims, dtype=np.float32) for _ in texts]

        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [np.array(item["embedding"], dtype=np.float32) for item in sorted_data]

    def get_embedding(self, text: str) -> "np.ndarray":
        """Legacy wrapper around embed."""
        return self.embed([text])[0]
