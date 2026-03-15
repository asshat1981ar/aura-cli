from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    import requests
except ImportError:  # pragma: no cover - exercised in subprocess-based CLI contract tests
    requests = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised in subprocess-based CLI contract tests
    np = None

from core.logging_utils import log_json # Import log_json
from core.file_tools import _aura_safe_loads # Import _aura_safe_loads
from core.config_manager import config
from core.runtime_auth import resolve_openai_api_key, resolve_openrouter_api_key
from core.token_utils import estimate_context_budget, compress_context
from core.model_cache import ModelCache


def _require_requests():
    if requests is None:
        raise ImportError(
            "ModelAdapter requires the optional 'requests' package for HTTP-backed model calls. "
            "Install it with `pip install requests` or `pip install -r requirements.txt`."
        )
    return requests


def _require_numpy():
    if np is None:
        raise ImportError(
            "ModelAdapter requires the optional 'numpy' package for embedding operations. "
            "Install it with `pip install numpy` or `pip install -r requirements.txt`."
        )
    return np

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
        Initializes the ModelAdapter, validates the Gemini CLI path,
        and defines an allowlist for executable tools.
        """
        # API keys will be fetched dynamically within their respective call methods
        self.gemini_cli_path = config.get("gemini_cli_path") # Configurable path to gemini CLI
        self.mcp_server_url = config.get("mcp_server_url") # Configurable MCP server URL
        self.router = None
        self.cache = None
        self.cache_db = None
        self._mcp_tool_bindings = config.get("mcp_tool_bindings", {}) or {}
        self._embedding_disabled = False
        self._embedding_disabled_reason = None
        self._embedding_disabled_logged = False
        self._local_profile_cooldowns: Dict[str, float] = {}
        
        # Configuration
        semantic_memory = config.get("semantic_memory", {}) or {}
        local_profiles = config.get("local_model_profiles", {}) or {}
        configured_embedding_model = semantic_memory.get("embedding_model") or "text-embedding-3-small"
        self._embedding_profile_name = None
        self._use_builtin_local_embeddings = False
        if isinstance(configured_embedding_model, str) and configured_embedding_model.startswith("openai/"):
            configured_embedding_model = configured_embedding_model.split("/", 1)[1]
        elif configured_embedding_model in {"local-tfidf-svd-50d", "local/tfidf-svd-50d"}:
            configured_embedding_model = "local-tfidf-svd-50d"
            self._use_builtin_local_embeddings = True
        elif isinstance(configured_embedding_model, str) and configured_embedding_model.startswith("local_profile:"):
            profile_name = configured_embedding_model.split(":", 1)[1]
            profile = local_profiles.get(profile_name, {}) if isinstance(local_profiles, dict) else {}
            if isinstance(profile, dict) and profile:
                self._embedding_profile_name = profile_name
                configured_embedding_model = profile.get("embedding_model") or profile.get("model") or configured_embedding_model
                self._embedding_dims = int(profile.get("embedding_dims") or 1536)
        self._embedding_model = configured_embedding_model # Default from config
        if not hasattr(self, "_embedding_dims"):
            self._embedding_dims = 50 if self._use_builtin_local_embeddings else 1536

        # Validate gemini CLI path
        if self.gemini_cli_path and not Path(self.gemini_cli_path).is_file():
            log_json("WARN", "gemini_cli_not_found", details={"path": self.gemini_cli_path})
            self.gemini_cli_path = None
        elif self.gemini_cli_path and not os.access(self.gemini_cli_path, os.X_OK):
            log_json("WARN", "gemini_cli_not_executable", details={"path": self.gemini_cli_path})
            self.gemini_cli_path = None

        # Define an explicit allowlist for tools
        self.ALLOWED_TOOLS = {
            "search", "read_file", "list_directory", "glob",
            # New GitHub tools
            "get_repo", "create_issue", "get_issue_details", "update_file", "get_pull_request_details"
        }

    @property
    def _mem_cache(self):
        """Backward compatibility for tests accessing internal cache."""
        cache = getattr(self, "cache", None)
        return cache.mem_cache if cache else {}

    @_mem_cache.setter
    def _mem_cache(self, value):
        """Backward compatibility setter."""
        cache = getattr(self, "cache", None)
        if cache:
            cache.mem_cache = value
        # If cache is None (e.g. in tests using __new__), we can't set it on the cache object.
        # But some tests might expect _mem_cache assignment to work even if cache is not enabled.
        # We might need to mock it or create a dummy cache if it's missing?
        # For now, let's just avoid the crash.


    def enable_cache(self, db_conn, ttl_seconds: int = 3600, momento=None):
        """Enables prompt-response caching.

        Args:
            db_conn:      SQLite connection (L2 persistent cache).
            ttl_seconds:  Cache TTL in seconds (default 1 hour).
            momento:      Optional :class:`MomentoAdapter` for L1 hot cache.
        """
        self.cache = ModelCache(db_conn, ttl_seconds, momento)
        self.cache_db = db_conn

    def preload_cache(self):
        """Loads the last 50 non-expired entries from the prompt_cache table into _mem_cache."""
        if self.cache:
            self.cache.preload()

    def set_router(self, router):
        """Attaches an adaptive router to the adapter."""
        self.router = router
        log_json("INFO", "model_router_attached")

    def _get_cached_response(self, prompt: str) -> str | None:
        if self.cache:
            return self.cache.get(prompt)
        return None

    def _save_to_cache(self, prompt: str, response: str):
        if self.cache:
            self.cache.set(prompt, response)

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        # [Tool execution logic unchanged]
        try:
            binding = self._mcp_tool_bindings.get(tool_name, {})
            if isinstance(binding, dict) and binding:
                server_name = binding.get("server")
                if not isinstance(server_name, str) or not server_name:
                    return f"MCP tool binding for {tool_name!r} is missing a valid server name."
                try:
                    port = config.get_mcp_server_port(server_name)
                except Exception as exc:
                    return f"MCP tool binding error for {tool_name!r}: {exc}"
                mcp_tool_url = f"http://localhost:{port}/call"
                requests_mod = _require_requests()
                try:
                    log_json(
                        "INFO",
                        "executing_tool_via_mcp_binding",
                        details={
                            "tool_name": tool_name,
                            "server": server_name,
                            "url": mcp_tool_url,
                            "args": args,
                        },
                    )
                    response = requests_mod.post(
                        mcp_tool_url,
                        json={"tool_name": tool_name, "args": args},
                        timeout=60,
                    )
                    response.raise_for_status() 
                    return json.dumps(response.json()) 
                except requests_mod.exceptions.RequestException as e:
                    return (
                        f"MCP server tool execution failed for {tool_name} "
                        f"via {server_name}: {e}"
                    )

            log_json(
                "INFO",
                "executing_tool_via_mcp_server",
                details={"tool_name": tool_name, "args": args},
            )
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

    def _make_request_with_retries(self, method, url, headers, json_payload, retries=3, backoff_factor=0.5):
        # [Retry logic unchanged]
        requests_mod = _require_requests()
        for attempt in range(retries):
            try:
                response = requests_mod.request(method, url, headers=headers, json=json_payload, timeout=60) 
                response.raise_for_status()
                return response
            except requests_mod.exceptions.RequestException as e:
                if attempt < retries - 1:
                    sleep_time = backoff_factor * (2 ** attempt)
                    log_json("WARN", "request_failed_retrying", details={"attempt": attempt + 1, "retries": retries, "error": str(e), "sleep_time": f"{sleep_time:.2f}"})
                    time.sleep(sleep_time)
                else:
                    raise 
        return None 

    # [LLM Call methods unchanged]
    def call_openrouter(self, prompt: str) -> str:
        openrouter_key = resolve_openrouter_api_key()
        if not openrouter_key:
            raise ValueError("OpenRouter-compatible API key not set for OpenRouter call.")
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "openrouter/auto",
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
                [self.gemini_cli_path, "--non-interactive", prompt],
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

    def call_openai(self, prompt: str) -> str:
        openai_api_key = resolve_openai_api_key()
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
        local_model_command = config.get("local_model_command")
        if not local_model_command:
            # Fallback for when local model is not configured - prevents crash/error loop
            log_json("WARN", "local_model_missing_using_fallback")
            return json.dumps({
                "status": "simulated_response",
                "message": "Local model not configured. Using fallback.",
                "code": "# Placeholder code from fallback\ndef recursive_self_improvement():\n    print('Simulated execution')"
            })

        try:
            command_parts = local_model_command.split()
            # If the command is a simple echo, we don't need to append the prompt in a way that breaks it
            # But for a real model, we usually pass the prompt.
            # Let's check if it's our dummy echo command
            if command_parts[0] == "echo":
                # For echo, we just run it as is, maybe ignoring the prompt to avoid shell injection if not careful
                # But the existing code appends prompt.
                # Let's just use shell=True for the fallback case in the config if needed,
                # but here we stick to the existing logic structure but make it robust.
                pass

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
            return json.dumps({"error": "Local model command not found."})
        except subprocess.CalledProcessError as e:
            return json.dumps({"error": f"Local model execution failed: {e.stderr}"})
        except Exception as e:
            return json.dumps({"error": f"Local model unexpected error: {str(e)}"})

    # Default timeout (seconds) for a single LLM call
    @property
    def LLM_TIMEOUT(self) -> int:
        return config.get("llm_timeout", 60)

    def _new_timeout_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Create a small executor for timeout-guarded model calls."""
        return concurrent.futures.ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="aura-llm",
        )

    def _call_with_timeout(
        self,
        fn,
        *args,
        timeout: int | None = None,
        executor: concurrent.futures.ThreadPoolExecutor | None = None,
    ) -> str:
        """Run *fn(*args)* in a thread and raise TimeoutError if it exceeds *timeout* seconds."""
        _timeout = timeout if timeout is not None else self.LLM_TIMEOUT
        owns_executor = executor is None
        timeout_executor = executor or self._new_timeout_executor()
        future = timeout_executor.submit(fn, *args)
        try:
            return future.result(timeout=_timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            log_json("ERROR", "llm_call_timeout",
                     details={"fn": fn.__name__, "timeout_s": _timeout})
            raise TimeoutError(
                f"LLM call to {fn.__name__!r} exceeded {_timeout}s timeout"
            ) from exc
        finally:
            if owns_executor:
                timeout_executor.shutdown(wait=False, cancel_futures=True)

    def respond(self, prompt: str):
        cached = self._get_cached_response(prompt)
        if cached:
            return cached
        model_response = None
        timeout_executor = self._new_timeout_executor()
        try:
            if self.router:
                try:
                    model_response = self._call_with_timeout(
                        self.router.route,
                        prompt,
                        executor=timeout_executor,
                    )
                except Exception as e:
                    log_json("WARN", "router_call_failed", details={"error": str(e), "fallback": "Direct fallbacks"})
            if not model_response:
                try:
                    model_response = self._call_with_timeout(
                        self.call_openai,
                        prompt,
                        executor=timeout_executor,
                    )
                except Exception as e:
                    log_json("WARN", "openai_call_failed", details={"error": str(e), "fallback": "OpenRouter"})
                    try:
                        model_response = self._call_with_timeout(
                            self.call_openrouter,
                            prompt,
                            executor=timeout_executor,
                        )
                    except Exception as e:
                        log_json("WARN", "openrouter_call_failed", details={"error": str(e), "fallback": "Local Model"})
                        model_response = self._call_with_timeout(
                            self.call_local,
                            prompt,
                            executor=timeout_executor,
                        )
        finally:
            timeout_executor.shutdown(wait=False, cancel_futures=True)
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

    def _get_local_model_profiles(self) -> Dict[str, Dict[str, Any]]:
        profiles = config.get("local_model_profiles", {}) or {}
        return profiles if isinstance(profiles, dict) else {}

    def _get_local_model_routing(self) -> Dict[str, str]:
        routing = config.get("local_model_routing", {}) or {}
        return routing if isinstance(routing, dict) else {}

    def _profile_on_cooldown(self, profile_name: str) -> bool:
        expires_at = self._local_profile_cooldowns.get(profile_name)
        return isinstance(expires_at, (int, float)) and expires_at > time.time()

    def _mark_profile_cooldown(self, profile_name: str, profile: Dict[str, Any]) -> None:
        cooldown_seconds = int(profile.get("cooldown_seconds", 30) or 30)
        self._local_profile_cooldowns[profile_name] = time.time() + cooldown_seconds

    def _ordered_profiles_for_role(self, role: str) -> List[str]:
        routing = self._get_local_model_routing()
        primary = routing.get(role)
        profiles = self._get_local_model_profiles()
        if not isinstance(primary, str) or primary not in profiles:
            return []

        ordered = [primary]
        profile = profiles.get(primary, {}) or {}
        for fallback in profile.get("fallback_profiles", []) or []:
            if isinstance(fallback, str) and fallback in profiles and fallback not in ordered:
                ordered.append(fallback)
        return ordered

    def _call_local_openai_compatible(self, profile: Dict[str, Any], prompt: str) -> str:
        base_url = str(profile.get("base_url") or "").rstrip("/")
        if not base_url:
            raise ValueError("Local OpenAI-compatible profile is missing base_url")
        url = f"{base_url}/chat/completions"
        payload = {
            "model": profile.get("model"),
            "messages": [{"role": "user", "content": prompt}],
        }
        if profile.get("temperature") is not None:
            payload["temperature"] = profile.get("temperature")
        if profile.get("max_tokens") is not None:
            payload["max_tokens"] = profile.get("max_tokens")
        response = self._make_request_with_retries("POST", url, {"Content-Type": "application/json"}, payload)
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _call_local_ollama(self, profile: Dict[str, Any], prompt: str) -> str:
        base_url = str(profile.get("base_url") or "http://127.0.0.1:11434").rstrip("/")
        url = f"{base_url}/api/generate"
        payload = {
            "model": profile.get("model"),
            "prompt": prompt,
            "stream": False,
            "options": {},
        }
        if profile.get("temperature") is not None:
            payload["options"]["temperature"] = profile.get("temperature")
        if profile.get("max_tokens") is not None:
            payload["options"]["num_predict"] = profile.get("max_tokens")
        response = self._make_request_with_retries("POST", url, {"Content-Type": "application/json"}, payload)
        data = response.json()
        return data["response"]

    def respond_for_role(self, role: str, prompt: str) -> str:
        profiles = self._get_local_model_profiles()
        ordered_profiles = self._ordered_profiles_for_role(role)
        if not ordered_profiles:
            return self.respond(prompt)

        for profile_name in ordered_profiles:
            profile = profiles.get(profile_name, {}) or {}
            if self._profile_on_cooldown(profile_name):
                continue
            provider = profile.get("provider")
            try:
                if provider == "openai_compatible":
                    return self._call_local_openai_compatible(profile, prompt)
                if provider == "ollama":
                    return self._call_local_ollama(profile, prompt)
            except Exception as exc:
                self._mark_profile_cooldown(profile_name, profile)
                log_json("WARN", "local_profile_failed", details={"profile": profile_name, "error": str(exc)})
                continue

        return self.respond(prompt)

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

        np_mod = _require_numpy()

        if self._use_builtin_local_embeddings:
            return [self._embed_builtin_local(text) for text in texts]

        if self._embedding_profile_name:
            profile = self._get_local_model_profiles().get(self._embedding_profile_name, {}) or {}
            if profile:
                return self._embed_local_profile(profile, texts)

        openai_api_key = resolve_openai_api_key()
        if not openai_api_key or self._embedding_disabled:
            reason = self._embedding_disabled_reason or "OPENAI_API_KEY not set for OpenAI embedding call."
            if not self._embedding_disabled_logged:
                log_json("WARN", "search_embedding_failed", details={"error": reason})
                self._embedding_disabled_logged = True
            # Return zero vectors — VectorStore search will rank all equally (no semantic search)
            return [np_mod.zeros(self._embedding_dims, dtype=np_mod.float32) for _ in texts]

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
            return [np_mod.zeros(self._embedding_dims, dtype=np_mod.float32) for _ in texts]
        
        # Sort by index to ensure order matches input
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [np_mod.array(item["embedding"], dtype=np_mod.float32) for item in sorted_data]

    def _embed_local_profile(self, profile: Dict[str, Any], texts: List[str]) -> List[np.ndarray]:
        np_mod = _require_numpy()
        provider = profile.get("provider")
        if provider != "openai_compatible":
            raise ValueError(f"Unsupported local embedding provider: {provider}")

        base_url = str(profile.get("base_url") or "").rstrip("/")
        if not base_url:
            raise ValueError("Local embedding profile is missing base_url")
        payload = {
            "input": texts,
            "model": profile.get("embedding_model") or profile.get("model"),
        }
        response = self._make_request_with_retries("POST", f"{base_url}/embeddings", {"Content-Type": "application/json"}, payload)
        data = response.json()
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [np_mod.array(item["embedding"], dtype=np_mod.float32) for item in sorted_data]

    def _embed_builtin_local(self, text: str) -> np.ndarray:
        np_mod = _require_numpy()
        vector = np_mod.zeros(self._embedding_dims, dtype=np_mod.float32)
        for token in str(text).lower().split():
            idx = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self._embedding_dims
            vector[idx] += 1.0
        norm = float(np_mod.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector

    def get_embedding(self, text: str) -> "np.ndarray":
        """Legacy wrapper around embed."""
        return self.embed([text])[0]
