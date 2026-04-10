"""Provider call methods for ModelAdapter (mixin).

Each ``call_*`` method talks to one LLM backend.  The mixin also owns the
shared ``_make_request_with_retries`` helper, local-profile infrastructure
(resolution, cooldowns, fallbacks), and local-embedding helpers that all
HTTP-based providers use.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from typing import Any, List

from core.logging_utils import log_json
from core.config_manager import config
from core.file_tools import _aura_safe_loads
from core.runtime_auth import (
    resolve_anthropic_api_key,
    resolve_local_model_profiles,
    resolve_openai_api_key,
    resolve_openrouter_api_key,
)

# Late-bound via model_adapter to keep patch targets stable.
import core.model_adapter as _ma


class ProvidersMixin:
    """Mixin providing individual LLM provider call methods."""

    # ------------------------------------------------------------------
    # HTTP retry helper
    # ------------------------------------------------------------------

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
        requests = _ma.requests
        for attempt in range(retries):
            try:
                response = requests.request(
                    method, url, headers=headers, json=json_payload, timeout=timeout,
                )
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

    # ------------------------------------------------------------------
    # OpenRouter
    # ------------------------------------------------------------------

    def call_openrouter(self, prompt: str, route_key: str | None = None) -> str:
        openrouter_key = resolve_openrouter_api_key() or os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise ValueError("OpenRouter-compatible API key not set for OpenRouter call.")

        import random
        routing = config.get("model_routing", {})
        selected_model = None
        if route_key:
            model_or_list = routing.get(route_key)
            if isinstance(model_or_list, list) and model_or_list:
                selected_model = random.choice(model_or_list)
            else:
                selected_model = model_or_list
        if not selected_model:
            selected_model = routing.get("fast", "google/gemini-2.0-flash-001")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/asshat1981ar/aura-cli",
            "X-Title": "AURA CLI",
        }
        payload = {
            "model": selected_model,
            "messages": [{"role": "user", "content": prompt}],
        }
        log_json("INFO", "openrouter_call", details={"model": selected_model, "route_key": route_key or "default"})
        response = self._make_request_with_retries("POST", url, headers, payload)
        data = response.json()

        if "error" in data:
            fallback = routing.get("fallback", "meta-llama/llama-3.3-70b-instruct:free")
            log_json("WARN", "openrouter_model_error", details={"model": selected_model, "error": str(data["error"]), "fallback": fallback})
            payload["model"] = fallback
            response = self._make_request_with_retries("POST", url, headers, payload)
            data = response.json()
            if "error" in data:
                raise ValueError(f"OpenRouter request failed (primary and fallback): {data['error']}")

        return data["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # Gemini CLI
    # ------------------------------------------------------------------

    def call_gemini(self, prompt: str) -> str:
        if not self.gemini_cli_path:
            raise ValueError("Gemini CLI path not configured or not executable.")
        try:
            result = subprocess.run(
                [self.gemini_cli_path, "-p", prompt, "--output-format", "text"],
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Gemini CLI call failed: {e.stderr}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling Gemini CLI: {e}") from e

    # ------------------------------------------------------------------
    # Codex CLI
    # ------------------------------------------------------------------

    def call_codex(self, prompt: str) -> str:
        if not self.codex_cli_path:
            raise ValueError("Codex CLI path not configured or not executable.")
        try:
            result = subprocess.run(
                [self.codex_cli_path, "exec", prompt, "--ask-for-approval", "never"],
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Codex CLI call failed: {e.stderr}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling Codex CLI: {e}") from e

    # ------------------------------------------------------------------
    # Copilot CLI
    # ------------------------------------------------------------------

    def call_copilot(self, prompt: str) -> str:
        if not self.copilot_cli_path:
            raise ValueError("Copilot CLI path not configured or not executable.")
        try:
            result = subprocess.run(
                [self.copilot_cli_path, "-p", prompt, "--silent", "--allow-all-tools"],
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Copilot CLI call failed: {e.stderr}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling Copilot CLI: {e}") from e

    # ------------------------------------------------------------------
    # OpenAI API
    # ------------------------------------------------------------------

    def call_openai(self, prompt: str) -> str:
        openai_api_key = resolve_openai_api_key() or os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not set for OpenAI call.")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
        }
        start = time.time()
        response = self._make_request_with_retries("POST", url, headers, payload)
        latency = time.time() - start
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        self._log_telemetry("openai", latency, content)
        return content

    # ------------------------------------------------------------------
    # Anthropic API
    # ------------------------------------------------------------------

    def call_anthropic(self, prompt: str) -> str:
        anthropic_api_key = resolve_anthropic_api_key() or os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set for Anthropic call.")
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "claude-3-5-sonnet-latest",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        start = time.time()
        response = self._make_request_with_retries("POST", url, headers, payload)
        latency = time.time() - start
        data = response.json()
        content = data["content"][0]["text"]
        self._log_telemetry("anthropic", latency, content)
        return content

    # ------------------------------------------------------------------
    # Local model (legacy command + profile-based)
    # ------------------------------------------------------------------

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
                    timeout=120,
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

    # ------------------------------------------------------------------
    # Local profile (with fallbacks and cooldowns)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Local profile provider dispatch
    # ------------------------------------------------------------------

    def _call_local_profile_provider(self, profile: dict, prompt: str) -> str:
        provider = str(profile.get("provider") or "openai_compatible")
        if provider == "openai_compatible":
            return self._call_local_openai_compatible(profile, prompt)
        if provider == "ollama":
            return self._call_local_ollama(profile, prompt)
        if provider == "command":
            return self._call_local_command_profile(profile, prompt)
        raise ValueError(f"Unsupported local model provider: {provider}")

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

        sanitized_prompt = shlex.quote(prompt) if prompt else ""
        use_stdin = True
        rendered_parts: list[str] = []
        for part in command_parts:
            if "{prompt}" in part:
                use_stdin = False
                rendered_parts.append(part.replace("{prompt}", sanitized_prompt))
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

    # ------------------------------------------------------------------
    # MCP / tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        requests = _ma.requests
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
                    timeout=60,
                )
                return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"Tool execution failed for {tool_name}: {e.stderr}"
        except json.JSONDecodeError:
            return f"Tool execution failed: Invalid JSON arguments for {tool_name}."
        except Exception as e:
            return f"Tool execution failed unexpectedly for {tool_name}: {str(e)}"

    # ------------------------------------------------------------------
    # Local profile resolution & health
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Local embedding helpers
    # ------------------------------------------------------------------

    def _call_local_openai_embeddings(self, profile: dict, texts: List) -> list:
        np = _ma.np
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

    def _call_local_command_embeddings(self, profile: dict, texts: List) -> list:
        np = _ma.np
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

    def _call_local_ollama_embeddings(self, profile: dict, texts: List) -> list:
        np = _ma.np
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
        except (AttributeError, IndexError, ValueError):
            pass

        vectors: list = []
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

    def _embed_with_local_profile(self, texts: List) -> list:
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
