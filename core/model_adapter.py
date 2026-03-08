from __future__ import annotations

import concurrent.futures
import hashlib
import os
import shlex
import subprocess
import json
import time
from pathlib import Path
from typing import Any, List


class _MissingPackage:
    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, item):
        raise ImportError(f"Optional dependency '{self._name}' is required for this feature. "
                          f"Install it with 'pip install {self._name}'.")

    def __call__(self, *args, **kwargs):
        raise ImportError(f"Optional dependency '{self._name}' is required for this feature. "
                          f"Install it with 'pip install {self._name}'.")

    def __repr__(self) -> str:
        return f"<missing package: {self._name}>"


try:
    import requests
except ImportError:
    requests = _MissingPackage("requests")

try:
    import numpy as np
except ImportError:
    np = _MissingPackage("numpy")

from core.logging_utils import log_json # Import log_json
from core.file_tools import _aura_safe_loads # Import _aura_safe_loads
from core.config_manager import config
from core.runtime_auth import resolve_local_model_profiles, resolve_openai_api_key, resolve_openrouter_api_key
from memory.embedding_provider import LocalEmbeddingProvider

# Removed dangerous global IPv4-only monkeypatch for socket.getaddrinfo.
# This monkeypatch forced all network connections to use IPv4, potentially
# causing connectivity issues and hiding underlying network misconfigurations.
# The system should now correctly handle both IPv4 and IPv6 as determined
# by the operating system and network stack.

class ModelAdapter:
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
        self.gemini_cli_path = config.get("gemini_cli_path") # Configurable path to gemini CLI
        self.codex_cli_path = config.get("codex_cli_path")   # Configurable path to codex CLI
        self.copilot_cli_path = config.get("copilot_cli_path") # Configurable path to copilot CLI
        self.mcp_server_url = config.get("mcp_server_url") # Configurable MCP server URL
        self.router = None
        self.cache_db = None
        self.cache_ttl = config.get("llm_timeout", 3600) # Use llm_timeout for cache as well if appropriate, or keep separate
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
        self._embedding_model = configured_embedding_model # Default from config
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
            # New GitHub tools
            "get_repo", "create_issue", "get_issue_details", "update_file", "get_pull_request_details"
        }

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

    def _get_local_profiles(self) -> dict[str, dict]:
        return resolve_local_model_profiles()

    def _get_local_routing(self) -> dict[str, str | None]:
        raw = config.get("local_model_routing", {}) or {}
        if not isinstance(raw, dict):
            return {}
        return {
            name: value
            for name, value in raw.items()
            if isinstance(name, str) and (value is None or isinstance(value, str))
        }

    def _resolve_local_profile_name(self, route_key: str) -> str | None:
        profile_name = self._get_local_routing().get(route_key)
        if isinstance(profile_name, str) and profile_name in self._get_local_profiles():
            return profile_name
        return None

    def _profile_timeout(self, profile: dict, *, key: str, default: float) -> float:
        try:
            value = float(profile.get(key, default))
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    def _profile_retries(self, profile: dict) -> int:
        try:
            value = int(profile.get("retries", 3))
        except (TypeError, ValueError):
            return 3
        return value if value > 0 else 3

    def _profile_backoff(self, profile: dict) -> float:
        try:
            value = float(profile.get("backoff_factor", 0.5))
        except (TypeError, ValueError):
            return 0.5
        return value if value >= 0 else 0.5

    def _profile_cooldown_seconds(self, profile: dict) -> float:
        return self._profile_timeout(profile, key="cooldown_seconds", default=30.0)

    def _profile_fallbacks(self, profile: dict) -> list[str]:
        fallback_profiles = profile.get("fallback_profiles")
        if isinstance(fallback_profiles, list):
            return [item for item in fallback_profiles if isinstance(item, str) and item.strip()]

        fallback_profile = profile.get("fallback_profile")
        if isinstance(fallback_profile, str) and fallback_profile.strip():
            return [fallback_profile]
        return []

    def _local_profile_cooldown_remaining(self, profile_name: str) -> float:
        until = self._local_profile_cooldowns.get(profile_name)
        if until is None:
            return 0.0
        return max(until - time.time(), 0.0)

    def _mark_local_profile_unhealthy(self, profile_name: str, profile: dict, exc: Exception) -> None:
        cooldown_seconds = self._profile_cooldown_seconds(profile)
        self._local_profile_cooldowns[profile_name] = time.time() + cooldown_seconds
        self._local_profile_cooldown_reasons[profile_name] = str(exc)
        log_json(
            "WARN",
            "local_profile_cooldown_started",
            details={
                "profile": profile_name,
                "cooldown_seconds": cooldown_seconds,
                "reason": str(exc),
            },
        )

    def _clear_local_profile_unhealthy(self, profile_name: str) -> None:
        self._local_profile_cooldowns.pop(profile_name, None)
        self._local_profile_cooldown_reasons.pop(profile_name, None)

    def _call_local_profile_provider(self, profile: dict, prompt: str) -> str:
        provider = str(profile.get("provider") or "openai_compatible")
        if provider == "openai_compatible":
            return self._call_local_openai_compatible(profile, prompt)
        if provider == "ollama":
            return self._call_local_ollama(profile, prompt)
        if provider == "command":
            return self._call_local_command_profile(profile, prompt)
        raise ValueError(f"Unsupported local model provider: {provider}")

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

    def _call_local_openai_compatible(self, profile: dict, prompt: str) -> str:
        base_url = str(profile.get("base_url") or "http://127.0.0.1:8080/v1").rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        url = f"{base_url}/chat/completions"
        model = profile.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Local openai_compatible profile requires a non-empty `model`.")

        headers = {"Content-Type": "application/json"}
        api_key = profile.get("api_key")
        if isinstance(api_key, str) and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(profile.get("temperature", 0.2)),
        }
        if profile.get("max_tokens") is not None:
            payload["max_tokens"] = int(profile["max_tokens"])
        if isinstance(profile.get("extra_body"), dict):
            payload.update(profile["extra_body"])

        response = self._make_request_with_retries(
            "POST",
            url,
            headers,
            payload,
            retries=self._profile_retries(profile),
            backoff_factor=self._profile_backoff(profile),
            timeout=self._profile_timeout(profile, key="request_timeout_seconds", default=float(self.LLM_TIMEOUT)),
            retry_label=f"local_openai:{model}",
        )
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _call_local_ollama(self, profile: dict, prompt: str) -> str:
        base_url = str(profile.get("base_url") or "http://127.0.0.1:11434").rstrip("/")
        url = f"{base_url}/api/generate"
        model = profile.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Local ollama profile requires a non-empty `model`.")

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        options: dict[str, Any] = {}
        if profile.get("temperature") is not None:
            options["temperature"] = float(profile["temperature"])
        if profile.get("max_tokens") is not None:
            options["num_predict"] = int(profile["max_tokens"])
        if options:
            payload["options"] = options
        if isinstance(profile.get("system"), str) and profile["system"].strip():
            payload["system"] = profile["system"]

        response = self._make_request_with_retries(
            "POST",
            url,
            {"Content-Type": "application/json"},
            payload,
            retries=self._profile_retries(profile),
            backoff_factor=self._profile_backoff(profile),
            timeout=self._profile_timeout(profile, key="request_timeout_seconds", default=float(self.LLM_TIMEOUT)),
            retry_label=f"local_ollama:{model}",
        )
        data = response.json()
        return data.get("response", "")

    def _call_local_command_profile(self, profile: dict, prompt: str) -> str:
        command = profile.get("command")
        if isinstance(command, str):
            command_parts = shlex.split(command)
        elif isinstance(command, list) and all(isinstance(part, str) for part in command):
            command_parts = list(command)
        else:
            raise ValueError("Local command profile requires `command` as a string or string list.")

        use_stdin = True
        rendered_parts: list[str] = []
        for part in command_parts:
            if "{prompt}" in part:
                use_stdin = False
                rendered_parts.append(part.replace("{prompt}", prompt))
            else:
                rendered_parts.append(part)

        result = subprocess.run(
            rendered_parts,
            input=prompt if use_stdin else None,
            capture_output=True,
            text=True,
            check=True,
            timeout=self._profile_timeout(profile, key="subprocess_timeout_seconds", default=120.0),
        )
        return result.stdout.strip()

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

        payload: dict[str, Any] = {
            "model": model,
            "input": texts,
        }
        response = self._make_request_with_retries(
            "POST",
            url,
            headers,
            payload,
            retries=self._profile_retries(profile),
            backoff_factor=self._profile_backoff(profile),
            timeout=self._profile_timeout(profile, key="request_timeout_seconds", default=float(self.LLM_TIMEOUT)),
            retry_label=f"local_openai_embeddings:{model}",
        )
        data = response.json()
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        vectors = [np.array(item["embedding"], dtype=np.float32) for item in sorted_data]
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
        vectors = [np.array(item, dtype=np.float32) for item in payload]
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
            response = self._make_request_with_retries(
                "POST",
                f"{base_url}/api/embed",
                headers,
                payload,
                retries=self._profile_retries(profile),
                backoff_factor=self._profile_backoff(profile),
                timeout=self._profile_timeout(profile, key="request_timeout_seconds", default=float(self.LLM_TIMEOUT)),
                retry_label=f"local_ollama_embeddings:{model}",
            )
            data = response.json()
            embeddings = data.get("embeddings")
            if isinstance(embeddings, list):
                vectors = [np.array(item, dtype=np.float32) for item in embeddings]
                if vectors:
                    self._embedding_dims = int(vectors[0].shape[0])
                return vectors
        except Exception:
            pass

        vectors: list[np.ndarray] = []
        for text in texts:
            response = self._make_request_with_retries(
                "POST",
                f"{base_url}/api/embeddings",
                headers,
                {"model": model, "prompt": text},
                retries=self._profile_retries(profile),
                backoff_factor=self._profile_backoff(profile),
                timeout=self._profile_timeout(profile, key="request_timeout_seconds", default=float(self.LLM_TIMEOUT)),
                retry_label=f"local_ollama_embeddings:{model}",
            )
            data = response.json()
            vectors.append(np.array(data["embedding"], dtype=np.float32))
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

    def call_local_profile(self, profile_name: str, prompt: str) -> str:
        return self._call_local_profile_with_fallbacks(profile_name, prompt, attempted_profiles=set())

    def _call_local_profile_with_fallbacks(
        self,
        profile_name: str,
        prompt: str,
        *,
        attempted_profiles: set[str],
    ) -> str:
        profiles = self._get_local_profiles()
        profile = profiles.get(profile_name)
        if not isinstance(profile, dict):
            raise ValueError(f"Unknown local model profile: {profile_name}")

        if profile_name in attempted_profiles:
            raise RuntimeError(f"Local profile fallback loop detected for {profile_name}.")
        attempted_profiles.add(profile_name)

        remaining = self._local_profile_cooldown_remaining(profile_name)
        if remaining > 0:
            for fallback_profile in self._profile_fallbacks(profile):
                if fallback_profile in attempted_profiles:
                    continue
                log_json(
                    "INFO",
                    "local_profile_fallback_due_to_cooldown",
                    details={
                        "profile": profile_name,
                        "fallback_profile": fallback_profile,
                        "remaining_seconds": round(remaining, 2),
                    },
                )
                return self._call_local_profile_with_fallbacks(
                    fallback_profile,
                    prompt,
                    attempted_profiles=attempted_profiles,
                )
            raise RuntimeError(
                f"Local profile '{profile_name}' is cooling down for another {remaining:.1f}s: "
                f"{self._local_profile_cooldown_reasons.get(profile_name, 'unknown error')}"
            )

        try:
            response = self._call_local_profile_provider(profile, prompt)
        except Exception as exc:
            self._mark_local_profile_unhealthy(profile_name, profile, exc)
            for fallback_profile in self._profile_fallbacks(profile):
                if fallback_profile in attempted_profiles:
                    continue
                log_json(
                    "WARN",
                    "local_profile_fallback_started",
                    details={
                        "profile": profile_name,
                        "fallback_profile": fallback_profile,
                        "error": str(exc),
                    },
                )
                return self._call_local_profile_with_fallbacks(
                    fallback_profile,
                    prompt,
                    attempted_profiles=attempted_profiles,
                )
            raise

        self._clear_local_profile_unhealthy(profile_name)
        return response

    # ... [Existing cache methods kept as is] ...
    def enable_cache(self, db_conn, ttl_seconds: int = 3600, momento=None):
        """Enables prompt-response caching.

        Args:
            db_conn:      SQLite connection (L2 persistent cache).
            ttl_seconds:  Cache TTL in seconds (default 1 hour).
            momento:      Optional :class:`MomentoAdapter` for L1 hot cache.
        """
        self.cache_db = db_conn
        self.cache_ttl = ttl_seconds
        self._momento = momento  # L1 cache adapter (may be None or unavailable)
        try:
            self.cache_db.execute("""
                CREATE TABLE IF NOT EXISTS prompt_cache (
                    prompt_hash TEXT PRIMARY KEY,
                    response TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.cache_db.commit()
            log_json("INFO", "model_cache_enabled", details={
                "ttl": ttl_seconds,
                "l1_momento": bool(momento and momento.is_available()),
            })
            self.preload_cache()
        except Exception as e:
            log_json("ERROR", "model_cache_init_failed", details={"error": str(e)})

    def preload_cache(self):
        """Loads the last 50 non-expired entries from the prompt_cache table into _mem_cache."""
        if not self.cache_db:
            return
        try:
            cursor = self.cache_db.execute(
                "SELECT prompt_hash, response FROM prompt_cache "
                "WHERE timestamp > datetime('now', ?) "
                "ORDER BY timestamp DESC LIMIT 50",
                (f"-{self.cache_ttl} seconds",)
            )
            rows = cursor.fetchall()
            for prompt_hash, response in rows:
                self._mem_cache[prompt_hash] = response
            log_json("INFO", "model_cache_preloaded", details={"count": len(rows)})
        except Exception as e:
            log_json("WARN", "model_cache_preload_failed", details={"error": str(e)})

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
        # Longer goals suggest more context is needed; add up to 2000 extra tokens
        extra = min(len(goal) // 10, 2000)
        return budget + extra

    def compress_context(self, text: str, max_tokens: int) -> str:
        """Truncates text to fit within max_tokens (rough estimate: 4 chars per token)."""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def set_router(self, router):
        """Attaches an adaptive router to the adapter."""
        self.router = router
        log_json("INFO", "model_router_attached")

    def _get_cached_response(self, prompt: str) -> str | None:
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

        # L0: in-memory dict (fastest)
        if prompt_hash in self._mem_cache:
            log_json("INFO", "model_cache_l0_hit", details={"prompt_hash": prompt_hash})
            return self._mem_cache[prompt_hash]

        # L1: Momento (sub-ms) — check before touching SQLite
        momento = getattr(self, "_momento", None)
        if momento and momento.is_available():
            try:
                from memory.momento_adapter import WORKING_MEMORY_CACHE
                key = f"response:{prompt_hash[:16]}"
                val = momento.cache_get(WORKING_MEMORY_CACHE, key)
                if val is not None:
                    log_json("INFO", "model_cache_l1_hit", details={"key": key})
                    return val
            except Exception as exc:
                log_json("WARN", "model_cache_l1_query_failed", details={"error": str(exc)})

        # L2: SQLite
        if not self.cache_db:
            return None
        try:
            cursor = self.cache_db.execute(
                "SELECT response FROM prompt_cache WHERE prompt_hash = ? AND timestamp > datetime('now', ?)",
                (prompt_hash, f"-{self.cache_ttl} seconds")
            )
            row = cursor.fetchone()
            if row:
                log_json("INFO", "model_cache_hit", details={"prompt_hash": prompt_hash})
                return row[0]
        except Exception as e:
            log_json("WARN", "model_cache_query_failed", details={"error": str(e)})
        return None

    def _save_to_cache(self, prompt: str, response: str):
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

        # L0: in-memory dict
        self._mem_cache[prompt_hash] = response

        # L1: Momento write-through
        momento = getattr(self, "_momento", None)
        if momento and momento.is_available():
            try:
                from memory.momento_adapter import WORKING_MEMORY_CACHE
                key = f"response:{prompt_hash[:16]}"
                momento.cache_set(WORKING_MEMORY_CACHE, key, response,
                                  ttl_seconds=self.cache_ttl)
            except Exception as exc:
                log_json("WARN", "model_cache_l1_save_failed", details={"error": str(exc)})

        # L2: SQLite
        if not self.cache_db:
            return
        try:
            self.cache_db.execute(
                "INSERT OR REPLACE INTO prompt_cache (prompt_hash, response, timestamp) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (prompt_hash, response)
            )
            self.cache_db.commit()
        except Exception as e:
            log_json("WARN", "model_cache_save_failed", details={"error": str(e)})

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        # [Tool execution logic unchanged]
        try:
            log_json("INFO", "executing_tool_via_mcp_server", details={"tool_name": tool_name, "args": args})
            if tool_name in ["get_repo", "create_issue", "get_issue_details", "update_file", "get_pull_request_details"]:
                mcp_tool_url = f"{self.mcp_server_url}/tool/{tool_name}"
                try:
                    response = requests.post(mcp_tool_url, json=args, timeout=60)
                    response.raise_for_status() 
                    return json.dumps(response.json()) 
                except requests.exceptions.RequestException as e:
                    return f"MCP Server tool execution failed for {tool_name}: {e}"
            else:
                args_str = json.dumps(args)
                command = ["npx", "@modelcontextprotocol/sdk", "call", tool_name, args_str]
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True, 
                    timeout=60
                )
                return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"Tool execution failed for {tool_name}: {e.stderr}"
        except json.JSONDecodeError:
            return f"Tool execution failed: Invalid JSON arguments for {tool_name}."
        except Exception as e:
            return f"Tool execution failed unexpectedly for {tool_name}: {str(e)}"

    def _make_request_with_retries(
        self,
        method,
        url,
        headers,
        json_payload,
        retries=3,
        backoff_factor=0.5,
        timeout=60,
        retry_label: str | None = None,
    ):
        # [Retry logic unchanged]
        for attempt in range(retries):
            try:
                response = requests.request(method, url, headers=headers, json=json_payload, timeout=timeout)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    sleep_time = backoff_factor * (2 ** attempt)
                    log_json(
                        "WARN",
                        "request_failed_retrying",
                        details={
                            "attempt": attempt + 1,
                            "retries": retries,
                            "error": str(e),
                            "sleep_time": f"{sleep_time:.2f}",
                            "timeout_s": timeout,
                            "label": retry_label,
                        },
                    )
                    time.sleep(sleep_time)
                else:
                    raise 
        return None 

    # [LLM Call methods unchanged]
    def call_openrouter(self, prompt: str) -> str:
        openrouter_key = resolve_openrouter_api_key() or os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise ValueError("OpenRouter-compatible API key not set for OpenRouter call.")

        # Free models to rotate through to avoid rate limits
        free_models = [
            "mistralai/mistral-7b-instruct:free",
            "google/gemini-2.0-flash-exp:free",
            "meta-llama/llama-3-8b-instruct:free",
            "qwen/qwen-2-7b-instruct:free"
        ]
        import random
        selected_model = random.choice(free_models)

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": selected_model,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = self._make_request_with_retries("POST", url, headers, payload)
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def call_gemini(self, prompt: str) -> str:
        if not self.gemini_cli_path:
            raise ValueError("Gemini CLI path not configured or not executable.")
        try:
            result = subprocess.run(
                [self.gemini_cli_path, "-p", prompt, "--output-format", "text"],
                capture_output=True,
                text=True,
                check=True,
                timeout=120
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Gemini CLI call failed: {e.stderr}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling Gemini CLI: {e}") from e

    def call_codex(self, prompt: str) -> str:
        if not self.codex_cli_path:
            raise ValueError("Codex CLI path not configured or not executable.")
        try:
            result = subprocess.run(
                [self.codex_cli_path, "exec", prompt, "--ask-for-approval", "never"],
                capture_output=True,
                text=True,
                check=True,
                timeout=120
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Codex CLI call failed: {e.stderr}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling Codex CLI: {e}") from e

    def call_copilot(self, prompt: str) -> str:
        if not self.copilot_cli_path:
            raise ValueError("Copilot CLI path not configured or not executable.")
        try:
            result = subprocess.run(
                [self.copilot_cli_path, "-p", prompt, "--silent", "--allow-all-tools"],
                capture_output=True,
                text=True,
                check=True,
                timeout=120
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Copilot CLI call failed: {e.stderr}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling Copilot CLI: {e}") from e

    def call_openai(self, prompt: str) -> str:
        openai_api_key = resolve_openai_api_key() or os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not set for OpenAI call.")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini", 
            "messages": [{"role": "user", "content": prompt}]
        }
        response = self._make_request_with_retries("POST", url, headers, payload)
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def call_local(self, prompt: str) -> str:
        default_profile = self._resolve_local_profile_name("fast")
        if default_profile:
            try:
                return self.call_local_profile(default_profile, prompt)
            except Exception as e:
                log_json("WARN", "local_profile_call_failed", details={"profile": default_profile, "error": str(e), "fallback": "legacy_local_model_command"})

        local_model_command = config.get("local_model_command")
        if local_model_command:
            try:
                command_parts = local_model_command.split()
                command_parts.append(prompt)
                result = subprocess.run(
                    command_parts,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=120 
                )
                return result.stdout.strip()
            except FileNotFoundError:
                return "Error: Local model command not found. Please ensure it's in your PATH or specify full path."
            except subprocess.CalledProcessError as e:
                return f"Error: Local model command failed with exit code {e.returncode}. Stderr: {e.stderr.strip()}"
            except Exception as e:
                return f"Error: An unexpected error occurred while calling local model: {e}"
        else:
            return "Local model not configured. Set 'local_model_command' in aura.config.json or AURA_LOCAL_MODEL_COMMAND env var."

    def respond_for_role(self, route_key: str, prompt: str) -> str:
        profile_name = self._resolve_local_profile_name(route_key)
        if not profile_name:
            return self.respond(prompt)

        cache_key = f"[local-role:{route_key}|profile:{profile_name}] {prompt}"
        cached = self._get_cached_response(cache_key)
        if cached:
            return cached

        try:
            model_response = self._call_with_timeout(self.call_local_profile, profile_name, prompt)
        except Exception as e:
            log_json("WARN", "local_role_call_failed", details={"route_key": route_key, "profile": profile_name, "error": str(e), "fallback": "default_model_path"})
            return self.respond(prompt)

        self._save_to_cache(cache_key, model_response)
        return model_response

    # Default timeout (seconds) for a single LLM call
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
            try:
                model_response = self._call_with_timeout(self.call_openai, prompt)
            except Exception as e:
                log_json("WARN", "openai_call_failed", details={"error": str(e), "fallback": "OpenRouter"})
                try:
                    model_response = self._call_with_timeout(self.call_openrouter, prompt)
                except Exception as e:
                    log_json("WARN", "openrouter_call_failed", details={"error": str(e), "fallback": "Local Model"})
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

    # --- EmbeddingProvider Interface ---

    def model_id(self) -> str:
        return self._embedding_model

    def dimensions(self) -> int:
        return self._embedding_dims

    def healthcheck(self) -> bool:
        """Checks if the embedding provider is reachable."""
        try:
            # Simple check with a dummy embedding
            self.embed(["test"])
            return True
        except Exception:
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
            # Return zero vectors — VectorStore search will rank all equally (no semantic search)
            return [np.zeros(self._embedding_dims, dtype=np.float32) for _ in texts]

        # Rate Limit Protection: Small sleep to avoid hitting burst limits
        import random
        time.sleep(0.2 + random.uniform(0, 0.1))

        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }
        
        # OpenAI supports batching
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
        
        # Sort by index to ensure order matches input
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [np.array(item["embedding"], dtype=np.float32) for item in sorted_data]

    def get_embedding(self, text: str) -> "np.ndarray":
        """Legacy wrapper around embed."""
        return self.embed([text])[0]
