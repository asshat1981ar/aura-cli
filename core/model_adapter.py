import hashlib
import os
import subprocess
import requests
import json
import time
from pathlib import Path
from typing import List
import numpy as np

from core.logging_utils import log_json # Import log_json
from core.file_tools import _aura_safe_loads # Import _aura_safe_loads

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
        self.gemini_cli_path = os.getenv("GEMINI_CLI_PATH", "/data/data/com.termux/files/usr/bin/gemini") # Configurable path to gemini CLI
        self.mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000") # Configurable MCP server URL
        self.router = None
        self.cache_db = None
        self.cache_ttl = 3600
        
        # Configuration
        self._embedding_model = "text-embedding-3-small" # Default from PRD
        self._embedding_dims = 1536

        # Validate gemini CLI path
        if not Path(self.gemini_cli_path).is_file():
            log_json("WARN", "gemini_cli_not_found", details={"path": self.gemini_cli_path})
            self.gemini_cli_path = None
        elif not os.access(self.gemini_cli_path, os.X_OK):
            log_json("WARN", "gemini_cli_not_executable", details={"path": self.gemini_cli_path})
            self.gemini_cli_path = None

        # In-memory cache (L0) — populated by preload_cache()
        self._mem_cache: dict = {}

        # Define an explicit allowlist for tools
        self.ALLOWED_TOOLS = {
            "search", "read_file", "list_directory", "glob",
            # New GitHub tools
            "get_repo", "create_issue", "get_issue_details", "update_file", "get_pull_request_details"
        }

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

    def _make_request_with_retries(self, method, url, headers, json_payload, retries=3, backoff_factor=0.5):
        # [Retry logic unchanged]
        for attempt in range(retries):
            try:
                response = requests.request(method, url, headers=headers, json=json_payload, timeout=60) 
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    sleep_time = backoff_factor * (2 ** attempt)
                    log_json("WARN", "request_failed_retrying", details={"attempt": attempt + 1, "retries": retries, "error": str(e), "sleep_time": f"{sleep_time:.2f}"})
                    time.sleep(sleep_time)
                else:
                    raise 
        return None 

    # [LLM Call methods unchanged]
    def call_openrouter(self, prompt: str) -> str:
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise ValueError("OPENROUTER_API_KEY not set for OpenRouter call.")
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
        openai_api_key = os.getenv("OPENAI_API_KEY")
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
        local_model_command = os.getenv("AURA_LOCAL_MODEL_COMMAND")
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
            return "Local model not configured. Set the AURA_LOCAL_MODEL_COMMAND environment variable " \
                   "to specify a command for local inference (e.g., 'ollama run llama2')."

    def respond(self, prompt: str):
        # [respond logic unchanged]
        cached = self._get_cached_response(prompt)
        if cached:
            return cached
        model_response = None
        if self.router:
            try:
                model_response = self.router.route(prompt)
            except Exception as e:
                log_json("WARN", "router_call_failed", details={"error": str(e), "fallback": "Direct fallbacks"})
        if not model_response:
            try:
                model_response = self.call_openai(prompt)
            except Exception as e:
                log_json("WARN", "openai_call_failed", details={"error": str(e), "fallback": "OpenRouter"})
                try:
                    model_response = self.call_openrouter(prompt)
                except Exception as e:
                    log_json("WARN", "openrouter_call_failed", details={"error": str(e), "fallback": "Local Model"})
                    model_response = self.call_local(prompt)
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
        Includes a small delay to avoid 429 Rate Limit errors.
        
        Args:
            texts: List of strings to embed.
            
        Returns:
            List of numpy arrays (float32).
        """
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not set for OpenAI embedding call.")

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

        response = self._make_request_with_retries("POST", url, headers, payload)
        data = response.json()
        
        # Sort by index to ensure order matches input
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [np.array(item["embedding"], dtype=np.float32) for item in sorted_data]

    def get_embedding(self, text: str) -> "np.ndarray":
        """Legacy wrapper around embed."""
        return self.embed([text])[0]
