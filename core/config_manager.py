import os
import json
import copy
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from core.logging_utils import log_json
from core.exceptions import ConfigurationError
from core.config_schema import ConfigValidator


def is_pydantic_available() -> bool:
    try:
        import pydantic  # noqa: F401

        return True
    except ImportError:
        return False


from core.credential_store import CredentialStore, get_credential_store

# ---------------------------------------------------------------------------
# Value validators — each returns (is_valid: bool, coerced_value, reason: str)
# ---------------------------------------------------------------------------


def _validate_positive_int(key: str, val: Any) -> Tuple[bool, Any, str]:
    try:
        v = int(val)
        if v > 0:
            return True, v, ""
        return False, None, f"{key} must be a positive integer, got {val!r}"
    except (ValueError, TypeError):
        return False, None, f"{key} must be an integer, got {val!r}"


def _validate_bool(key: str, val: Any) -> Tuple[bool, Any, str]:
    if isinstance(val, bool):
        return True, val, ""
    if isinstance(val, str) and val.lower() in ("true", "false", "1", "0", "yes", "no"):
        return True, val.lower() in ("true", "1", "yes"), ""
    return False, None, f"{key} must be a boolean, got {val!r}"


def _validate_string(key: str, val: Any) -> Tuple[bool, Any, str]:
    if val is None or isinstance(val, str):
        return True, val, ""
    return False, None, f"{key} must be a string, got {val!r}"


def _validate_float_range(key: str, val: Any, lo: float, hi: float) -> Tuple[bool, Any, str]:
    try:
        v = float(val)
        if lo <= v <= hi:
            return True, v, ""
        return False, None, f"{key} must be in [{lo}, {hi}], got {val!r}"
    except (ValueError, TypeError):
        return False, None, f"{key} must be a number, got {val!r}"


# Key → validator function (None = no validation, just pass through)
_KEY_VALIDATORS = {
    "max_iterations": lambda k, v: _validate_positive_int(k, v),
    "max_cycles": lambda k, v: _validate_positive_int(k, v),
    "policy_max_cycles": lambda k, v: _validate_positive_int(k, v),
    "policy_max_seconds": lambda k, v: _validate_positive_int(k, v),
    "dry_run": lambda k, v: _validate_bool(k, v),
    "decompose": lambda k, v: _validate_bool(k, v),
    "strict_schema": lambda k, v: _validate_bool(k, v),
    "model_name": lambda k, v: _validate_string(k, v),
    "api_key": lambda k, v: _validate_string(k, v),
    "openai_api_key": lambda k, v: _validate_string(k, v),
    "anthropic_api_key": lambda k, v: _validate_string(k, v),
    "memory_store_path": lambda k, v: _validate_string(k, v),
    "brain_db_path": lambda k, v: _validate_string(k, v),
    "goal_queue_path": lambda k, v: _validate_string(k, v),
    "memory_persistence_path": lambda k, v: _validate_string(k, v),
    "auto_add_capabilities": lambda k, v: _validate_bool(k, v),
    "auto_queue_missing_capabilities": lambda k, v: _validate_bool(k, v),
    "auto_provision_mcp": lambda k, v: _validate_bool(k, v),
    "auto_start_mcp_servers": lambda k, v: _validate_bool(k, v),
    "auto_backfill_coverage": lambda k, v: _validate_bool(k, v),
    "enable_mcp_registry": lambda k, v: _validate_bool(k, v),
    "enable_new_orchestrator": lambda k, v: _validate_bool(k, v),
    "force_legacy_orchestrator": lambda k, v: _validate_bool(k, v),
    "new_orchestrator_shadow_mode": lambda k, v: _validate_bool(k, v),
    "reliability_threshold": lambda k, v: _validate_float_range(k, v, 0.0, 100.0),
    "gemini_cli_path": lambda k, v: _validate_string(k, v),
    "codex_cli_path": lambda k, v: _validate_string(k, v),
    "copilot_cli_path": lambda k, v: _validate_string(k, v),
    "mcp_server_url": lambda k, v: _validate_string(k, v),
    "local_model_command": lambda k, v: _validate_string(k, v),
    "llm_timeout": lambda k, v: _validate_positive_int(k, v),
}

DEFAULT_CONFIG = {
    "model_name": "google/gemini-2.0-flash-exp:free",
    "api_key": None,
    "openai_api_key": None,
    "anthropic_api_key": None,
    "dry_run": False,
    "decompose": False,
    "max_iterations": 10,
    "max_cycles": 5,
    "strict_schema": False,
    "policy_name": "sliding_window",
    "policy_max_cycles": 5,
    "policy_max_seconds": 120,
    "memory_persistence_path": "memory/task_hierarchy_v2.json",
    "memory_store_path": "memory/store",
    # R8: standardised path — no v2 suffix
    "goal_queue_path": "memory/goal_queue.json",
    "goal_archive_path": "memory/goal_archive_v2.json",
    "brain_db_path": "memory/brain_v2.db",
    "auto_add_capabilities": True,
    "auto_queue_missing_capabilities": True,
    "auto_provision_mcp": False,
    "auto_start_mcp_servers": False,
    "auto_backfill_coverage": True,
    "enable_mcp_registry": True,
    "enable_new_orchestrator": True,
    "force_legacy_orchestrator": False,
    "new_orchestrator_shadow_mode": False,
    "reliability_threshold": 80.0,
    "gemini_cli_path": "/data/data/com.termux/files/usr/bin/gemini",
    "codex_cli_path": "/data/data/com.termux/files/usr/bin/codex",
    "copilot_cli_path": "/data/data/com.termux/files/usr/bin/copilot",
    "mcp_server_url": "http://localhost:8000",
    "local_model_command": None,
    "local_model_profiles": {},
    "local_model_routing": {
        "planning": None,
        "analysis": None,
        "critique": None,
        "code_generation": None,
        "embedding": None,
        "quality": None,
        "fast": None,
    },
    "llm_timeout": 60,
    "beads": {
        "enabled": True,
        "required": True,
        "bridge_command": None,
        "timeout_seconds": 20,
        "scope": "goal_run",
        "persist_artifacts": True,
    },
    "model_routing": {
        "code_generation": "google/gemini-2.0-flash-exp:free",
        "planning": "google/gemini-2.0-flash-exp:free",
        "analysis": "google/gemini-2.0-flash-exp:free",
        "critique": "google/gemini-2.0-flash-exp:free",
        "embedding": "openai/text-embedding-3-small",
        "fast": "google/gemini-2.0-flash-exp:free",
        "quality": "anthropic/claude-3.5-sonnet",
    },
    # R4: MCP server port registry
    "mcp_servers": {
        "dev_tools": 8001,
        "skills": 8002,
        "control": 8003,
        "agentic_loop": 8006,
        "copilot": 8007,
    },
    # R8: MCP server API keys (optional, for authenticated access)
    "mcp_server_api_keys": {
        "dev_tools": None,
        "skills": None,
        "control": None,
        "agentic_loop": None,
        "copilot": None,
    },
    # ASCM v2 Configuration
    "semantic_memory": {"enabled": True, "backend": "sqlite_local", "embedding_model": "text-embedding-3-small", "top_k": 10, "min_score": 0.65, "max_snippet_chars": 2000, "reindex_on_model_change": True, "eval_sampling_rate": 0.1},
}


class ConfigManager:
    """
    Unified Control Plane: Centralized configuration manager for AURA.
    Enforces a tiered strategy: (Overrides > ENV > Credential Store > JSON > Defaults).

    Security Issue #427: API keys are now retrieved from secure credential store
    (OS keyring) when available, falling back to environment variables and config file.
    """

    # Keys that should be retrieved from secure credential store
    SECURE_CONFIG_KEYS = {
        "api_key",
        "openai_api_key",
        "anthropic_api_key",
        "github_token",
        "azure_api_key",
    }

    def __init__(
        self,
        config_file=None,
        overrides: Optional[Dict[str, Any]] = None,
        credential_store: Optional[CredentialStore] = None,
    ):
        # Look for settings.json first, then aura.config.json
        if config_file:
            self.config_file = Path(config_file)
        elif Path("settings.json").exists():
            self.config_file = Path("settings.json")
        else:
            self.config_file = Path("aura.config.json")

        self.runtime_overrides = overrides or {}
        self.file_config = {}
        self.effective_config = {}

        # Initialize credential store for secure config values
        self._credential_store = credential_store or get_credential_store()

        self.refresh()

    def _load_from_file(self) -> Dict[str, Any]:
        if not self.config_file.exists():
            return {}
        try:
            with open(self.config_file, "r") as f:
                data = json.load(f)
                log_json("INFO", "config_loaded_from_file", details={"path": str(self.config_file)})

                # If using the new settings.json format, the AURA config is under the "aura" key
                if "aura" in data and isinstance(data["aura"], dict):
                    # Flatten the aura section for the ConfigManager
                    aura_config = data["aura"]

                    # Map new optimized keys back to flat keys if necessary
                    if "context_management" in aura_config:
                        aura_config["semantic_memory"] = aura_config.pop("context_management")

                    return aura_config

                return data
        except json.JSONDecodeError as e:
            log_json("ERROR", "config_parse_failed", details={"error": str(e)})
            raise ConfigurationError(f"Failed to parse config file: {e}")

    def _load_from_env(self) -> Dict[str, Any]:
        env_config = {}

        # Legacy mappings and direct ENV overrides
        env_mappings = {
            "OPENROUTER_API_KEY": "api_key",
            "AURA_API_KEY": "api_key",
            "OPENAI_API_KEY": "openai_api_key",
            "ANTHROPIC_API_KEY": "anthropic_api_key",
            "GEMINI_CLI_PATH": "gemini_cli_path",
            "CODEX_CLI_PATH": "codex_cli_path",
            "COPILOT_CLI_PATH": "copilot_cli_path",
            "MCP_SERVER_URL": "mcp_server_url",
            "AURA_LOCAL_MODEL_COMMAND": "local_model_command",
            "AURA_LLM_TIMEOUT": "llm_timeout",
            "AURA_ENABLE_MCP_REGISTRY": "enable_mcp_registry",
            "AURA_ENABLE_NEW_ORCHESTRATOR": "enable_new_orchestrator",
            "AURA_FORCE_LEGACY_ORCHESTRATOR": "force_legacy_orchestrator",
            "AURA_NEW_ORCHESTRATOR_SHADOW_MODE": "new_orchestrator_shadow_mode",
        }

        for env_key, config_key in env_mappings.items():
            if env_key in os.environ:
                env_config[config_key] = os.environ[env_key]

        # Standard AURA_* overrides for all keys in DEFAULT_CONFIG
        for key in DEFAULT_CONFIG:
            env_key = f"AURA_{key.upper()}"
            if env_key in os.environ:
                val = os.environ[env_key]
                # Type coercion with error handling
                try:
                    if isinstance(DEFAULT_CONFIG[key], bool):
                        env_config[key] = val.lower() in ("true", "1", "yes")
                    elif isinstance(DEFAULT_CONFIG[key], int):
                        env_config[key] = int(val)
                    else:
                        env_config[key] = val
                except (ValueError, TypeError):
                    log_json("WARN", "config_env_coercion_failed", details={"key": key, "val": val})
                    # Skip this key, let it fall back to JSON/Default
                    continue

        # Handle nested model_routing overrides
        routing_overrides = {}
        for sub_key in ["code_generation", "planning", "analysis", "critique", "embedding", "fast", "quality"]:
            env_name = f"AURA_MODEL_ROUTING_{sub_key.upper()}"
            if env_name in os.environ:
                routing_overrides[sub_key] = os.environ[env_name]
        if routing_overrides:
            env_config["model_routing"] = {**DEFAULT_CONFIG["model_routing"], **routing_overrides}

        # Handle nested local_model_routing overrides
        local_routing_overrides = {}
        for sub_key in DEFAULT_CONFIG["local_model_routing"]:
            env_name = f"AURA_LOCAL_MODEL_ROUTING_{sub_key.upper()}"
            if env_name in os.environ:
                local_routing_overrides[sub_key] = os.environ[env_name]
        if local_routing_overrides:
            env_config["local_model_routing"] = {
                **DEFAULT_CONFIG["local_model_routing"],
                **local_routing_overrides,
            }

        # Handle nested semantic_memory overrides
        sem_mem_overrides = {}
        for sub_key, default_val in DEFAULT_CONFIG["semantic_memory"].items():
            env_name = f"AURA_SEMANTIC_MEMORY_{sub_key.upper()}"
            if env_name in os.environ:
                val = os.environ[env_name]
                if isinstance(default_val, bool):
                    sem_mem_overrides[sub_key] = val.lower() in ("true", "1", "yes")
                elif isinstance(default_val, int):
                    sem_mem_overrides[sub_key] = int(val)
                elif isinstance(default_val, float):
                    sem_mem_overrides[sub_key] = float(val)
                else:
                    sem_mem_overrides[sub_key] = val
        if sem_mem_overrides:
            env_config["semantic_memory"] = {**DEFAULT_CONFIG["semantic_memory"], **sem_mem_overrides}

        # R5: Handle nested mcp_servers port overrides
        # Supports AURA_MCP_SERVERS_<SERVER_NAME>_PORT for each server
        mcp_port_overrides = {}
        default_mcp_servers = DEFAULT_CONFIG.get("mcp_servers", {})
        for server_name in default_mcp_servers:
            env_name = f"AURA_MCP_SERVERS_{server_name.upper()}_PORT"
            if env_name in os.environ:
                try:
                    mcp_port_overrides[server_name] = int(os.environ[env_name])
                except (ValueError, TypeError):
                    log_json("WARN", "config_mcp_port_coercion_failed", details={"server": server_name, "val": os.environ[env_name]})
        if mcp_port_overrides:
            env_config["mcp_servers"] = {**default_mcp_servers, **mcp_port_overrides}

        return env_config

    def refresh(self):
        """Re-evaluates the effective configuration based on the tier hierarchy."""
        self.file_config = self._load_from_file()
        env_config = self._load_from_env()

        # Merge hierarchy: Defaults < JSON < ENV < Overrides
        merged = copy.deepcopy(DEFAULT_CONFIG)

        # Deep merge for nested config groups to preserve defaults if partial override
        if "beads" in self.file_config:
            merged["beads"].update(self.file_config["beads"])
        if "model_routing" in self.file_config:
            merged["model_routing"].update(self.file_config["model_routing"])
        if "semantic_memory" in self.file_config:
            merged["semantic_memory"].update(self.file_config["semantic_memory"])
        if "local_model_profiles" in self.file_config:
            merged["local_model_profiles"].update(self.file_config["local_model_profiles"])
        if "local_model_routing" in self.file_config:
            merged["local_model_routing"].update(self.file_config["local_model_routing"])
        if "mcp_servers" in self.file_config:
            merged["mcp_servers"].update(self.file_config["mcp_servers"])
        if "mcp_server_api_keys" in self.file_config:
            merged["mcp_server_api_keys"].update(self.file_config["mcp_server_api_keys"])

        # Update other top-level keys
        for k, v in self.file_config.items():
            if k not in ["beads", "model_routing", "semantic_memory", "local_model_profiles", "local_model_routing", "mcp_servers", "mcp_server_api_keys"]:
                merged[k] = v

        # Merge ENV (env_config already has merged sub-dicts)
        if "beads" in env_config:
            merged["beads"].update(env_config["beads"])
        if "model_routing" in env_config:
            merged["model_routing"].update(env_config["model_routing"])
        if "semantic_memory" in env_config:
            merged["semantic_memory"].update(env_config["semantic_memory"])
        if "local_model_routing" in env_config:
            merged["local_model_routing"].update(env_config["local_model_routing"])
        if "mcp_servers" in env_config:
            merged["mcp_servers"].update(env_config["mcp_servers"])
        if "mcp_server_api_keys" in env_config:
            merged["mcp_server_api_keys"].update(env_config["mcp_server_api_keys"])

        for k, v in env_config.items():
            if k not in ["beads", "model_routing", "semantic_memory", "local_model_routing", "mcp_servers", "mcp_server_api_keys"]:
                merged[k] = v

        merged.update(self.runtime_overrides)

        # P1 FIX: Pydantic schema validation at startup
        self._validate_with_schema(merged)

        self.effective_config = merged

    def _validate_with_schema(self, config: Dict[str, Any]) -> None:
        """Validate configuration using Pydantic schema.

        Logs warnings for invalid values but doesn't raise exceptions
        to maintain backward compatibility.
        """
        validator = ConfigValidator()
        is_valid, validated, errors = validator.validate(config)

        if not is_valid:
            for error in errors:
                log_json("WARN", "config_validation_failed", details={"error": error})
            if is_pydantic_available():
                log_json("INFO", "config_using_defaults_for_invalid")
        elif is_pydantic_available():
            log_json("DEBUG", "config_validated_with_pydantic")

    def _validate_value(self, key: str, value: Any) -> Any:
        """Validate *value* for *key*; return coerced value or DEFAULT_CONFIG fallback on error."""
        validator = _KEY_VALIDATORS.get(key)
        if validator is None:
            return value
        ok, coerced, reason = validator(key, value)
        if ok:
            return coerced
        default = DEFAULT_CONFIG.get(key)
        log_json("ERROR", "config_value_invalid", details={"key": key, "value": value, "reason": reason, "fallback": default})
        return default

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value from the effective config.

        For sensitive keys (api_key, openai_api_key, anthropic_api_key, etc.),
        checks the secure credential store first before falling back to config.

        Priority order:
        1. Runtime overrides
        2. Environment variables
        3. Secure credential store (for sensitive keys)
        4. Config file
        5. Default values

        Security Issue #427: Secure credential storage integration
        """
        # Check for sensitive keys in credential store
        if key in self.SECURE_CONFIG_KEYS:
            # Priority 1: Runtime overrides
            if key in self.runtime_overrides:
                return self._validate_value(key, self.runtime_overrides[key]) if key in _KEY_VALIDATORS else self.runtime_overrides[key]

            # Priority 2: Environment variables
            env_key = f"AURA_{key.upper()}"
            env_value = os.environ.get(env_key) or os.environ.get(key.upper())
            if env_value:
                return self._validate_value(key, env_value) if key in _KEY_VALIDATORS else env_value

            # Priority 3: Secure credential store
            credential_value = self._credential_store.retrieve(key)
            if credential_value:
                return self._validate_value(key, credential_value) if key in _KEY_VALIDATORS else credential_value

            # Priority 4: Config file / defaults
            val = self.effective_config.get(key, default)
            return self._validate_value(key, val) if key in _KEY_VALIDATORS else val

        # Non-sensitive keys: use normal retrieval
        val = self.effective_config.get(key, default)
        return self._validate_value(key, val) if key in _KEY_VALIDATORS else val

    def show_config(self) -> Dict[str, Any]:
        """Return the effective config dict (for --show-config / diagnostics)."""
        return dict(self.effective_config)

    def set_runtime_override(self, key: str, value: Any):
        """Sets a temporary runtime override."""
        self.runtime_overrides[key] = value
        self.refresh()

    def update_config(self, updates: Dict[str, Any], persist: bool = True):
        """Update multiple configuration values, optionally persisting to disk.

        Supports deep merging for nested config groups.
        """
        for k, v in updates.items():
            if k in ["beads", "model_routing", "semantic_memory", "local_model_profiles", "local_model_routing", "mcp_server_api_keys"] and isinstance(v, dict):
                if k not in self.file_config:
                    self.file_config[k] = {}
                self.file_config[k].update(v)
            else:
                self.file_config[k] = v

        if persist:
            try:
                with open(self.config_file, "w") as f:
                    json.dump(self.file_config, f, indent=4)
                log_json("INFO", "config_updated_and_persisted", details={"keys": list(updates.keys())})
            except Exception as e:
                log_json("ERROR", "config_save_failed", details={"error": str(e)})
                raise ConfigurationError(f"Failed to save config: {e}")

        self.refresh()

    def persist_to_file(self, key: str, value: Any):
        """Sets a configuration value and persists it to the aura.config.json file."""
        self.file_config[key] = value
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.file_config, f, indent=4)
            log_json("INFO", "config_persisted", details={"key": key})
            self.refresh()
        except Exception as e:
            log_json("ERROR", "config_save_failed", details={"error": str(e)})
            raise ConfigurationError(f"Failed to save config: {e}")

    def get_mcp_server_port(self, server_name: str) -> int:
        """R4: Return the configured port for a named MCP server.

        Falls back to the DEFAULT_CONFIG registry when the key is absent from
        the effective config.  Raises ``ConfigurationError`` for unknown names.

        Args:
            server_name: One of ``dev_tools``, ``skills``, ``control``,
                         ``agentic_loop``, or ``copilot``.

        Returns:
            TCP port number (int).
        """
        registry: dict = self.effective_config.get("mcp_servers", {})
        if server_name not in registry:
            default_registry = DEFAULT_CONFIG.get("mcp_servers", {})
            if server_name not in default_registry:
                raise ConfigurationError(f"Unknown MCP server name '{server_name}'. Known servers: {list(default_registry.keys())}")
            return int(default_registry[server_name])
        return int(registry[server_name])

    def get_mcp_server_api_key(self, server_name: str) -> Optional[str]:
        """R8: Return the configured API key for a named MCP server.

        Falls back to environment variables following the pattern:
        - MCP_<SERVER_NAME>_API_KEY (e.g., MCP_DEV_TOOLS_API_KEY)
        - Legacy: MCP_API_TOKEN for dev_tools

        Args:
            server_name: One of ``dev_tools``, ``skills``, ``control``,
                         ``agentic_loop``, or ``copilot``.

        Returns:
            API key string or None if not configured.

        Raises:
            ConfigurationError: For unknown server names.
        """
        # Validate server name
        default_registry = DEFAULT_CONFIG.get("mcp_servers", {})
        if server_name not in default_registry:
            raise ConfigurationError(f"Unknown MCP server name '{server_name}'. Known servers: {list(default_registry.keys())}")

        # 1. Check environment variable (highest priority)
        env_var = f"MCP_{server_name.upper()}_API_KEY"
        env_key = os.getenv(env_var, "").strip()
        if env_key:
            return env_key

        # 2. Check config file
        key_registry: dict = self.effective_config.get("mcp_server_api_keys", {})
        cfg_key = key_registry.get(server_name)
        if cfg_key:
            return cfg_key

        # 3. Legacy fallback for dev_tools
        if server_name == "dev_tools":
            legacy_token = os.getenv("MCP_API_TOKEN", "").strip()
            if legacy_token:
                return legacy_token

        return None

    def check_port_available(self, port: int, host: str = "0.0.0.0", strict: bool = False) -> bool:
        """Check if a TCP port is available for binding.

        Args:
            port: The port number to check.
            host: The host address to bind to (default: 0.0.0.0).
            strict: If True, performs a more strict check that detects more
                   conflicts but may have false positives on some systems.

        Returns:
            True if the port is available, False otherwise.
        """
        # Validate port range
        if not isinstance(port, int) or not (0 <= port <= 65535):
            return False

        # First try without SO_REUSEADDR for accurate conflict detection
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                # Don't set SO_REUSEADDR to detect actual conflicts
                sock.bind((host, port))
                return True
        except (OSError, socket.error):
            return False

    def check_mcp_server_port_conflicts(self, servers: Optional[List[str]] = None, raise_on_conflict: bool = False) -> Dict[str, Dict[str, Any]]:
        """Check for port conflicts among MCP servers.

        R5: Port Conflict Detection - scans configured ports and reports
        any that are already in use, helping prevent startup failures.

        Args:
            servers: List of server names to check. If None, checks all
                     servers in the mcp_servers config.
            raise_on_conflict: If True, raises ConfigurationError when
                               conflicts are detected.

        Returns:
            Dict mapping server_name -> {
                "port": int,
                "available": bool,
                "error": Optional[str]
            }

        Raises:
            ConfigurationError: If raise_on_conflict=True and conflicts exist.
        """
        registry: Dict[str, int] = self.effective_config.get("mcp_servers", {})
        if not registry:
            registry = DEFAULT_CONFIG.get("mcp_servers", {})

        servers_to_check = servers or list(registry.keys())
        results: Dict[str, Dict[str, Any]] = {}
        conflicts: List[str] = []

        for server_name in servers_to_check:
            if server_name not in registry:
                error_msg = f"Unknown MCP server: {server_name}"
                results[server_name] = {"port": None, "available": False, "error": error_msg}
                conflicts.append(error_msg)
                continue

            port = int(registry[server_name])
            is_available = self.check_port_available(port)

            if not is_available:
                error_msg = f"Port {port} for '{server_name}' is already in use. Consider: 1) Stop the other service, 2) Configure a different port in aura.config.json mcp_servers.{server_name}, or 3) Set the appropriate env var override."
                results[server_name] = {"port": port, "available": False, "error": error_msg}
                conflicts.append(error_msg)
                log_json("WARN", "mcp_port_conflict_detected", details={"server": server_name, "port": port})
            else:
                results[server_name] = {"port": port, "available": True, "error": None}

        if conflicts and raise_on_conflict:
            raise ConfigurationError(f"MCP server port conflicts detected ({len(conflicts)}): " + "; ".join(conflicts))

        return results

    def find_available_port(self, start_port: int = 8000, end_port: int = 9000, exclude_ports: Optional[Set[int]] = None) -> Optional[int]:
        """Find an available TCP port in a range.

        Args:
            start_port: Start of port range (inclusive).
            end_port: End of port range (inclusive).
            exclude_ports: Set of ports to skip even if available.

        Returns:
            First available port, or None if none found.
        """
        exclude = exclude_ports or set()
        for port in range(start_port, end_port + 1):
            if port in exclude:
                continue
            if self.check_port_available(port):
                return port
        return None

    def suggest_mcp_port_reassignments(self) -> Dict[str, int]:
        """Suggest alternative ports for any configured ports that are in use.

        Returns:
            Dict mapping server_name -> suggested_port for any servers
            with port conflicts. Empty dict if no conflicts.
        """
        conflicts = self.check_mcp_server_port_conflicts()
        suggestions: Dict[str, int] = {}

        # Track ports we've already suggested to avoid duplicates
        used_ports: Set[int] = set()

        for server_name, info in conflicts.items():
            if not info["available"]:
                # Find a new port starting from the default range
                suggested = self.find_available_port(start_port=8001, end_port=8100, exclude_ports=used_ports)
                if suggested:
                    suggestions[server_name] = suggested
                    used_ports.add(suggested)
                    log_json("INFO", "mcp_port_suggestion", details={"server": server_name, "suggested_port": suggested})

        return suggestions

    def bootstrap(self):
        """Generates a default aura.config.json if it doesn't exist."""
        if self.config_file.exists():
            log_json("INFO", "config_bootstrap_skipped_exists")
            return

        # Create minimal valid config
        bootstrap_data = {"model_name": DEFAULT_CONFIG["model_name"], "api_key": "YOUR_OPENROUTER_API_KEY_HERE"}

        with open(self.config_file, "w") as f:
            json.dump(bootstrap_data, f, indent=4)
        log_json("INFO", "config_bootstrapped", details={"path": str(self.config_file)})

    def interactive_bootstrap(self):
        """Interactively guides the user through setting up AURA configuration."""
        print("\n--- AURA Interactive Bootstrap ---")
        print("This will create or update your aura.config.json file.")
        print("Note: API keys can be stored securely in your OS keyring.")

        try:
            # Ask about secure storage preference
            use_secure = input("Store API keys in secure credential store? [Y/n]: ").strip().lower()
            store_securely = use_secure in ("", "y", "yes")

            # 1. API Key
            current_key = self.get("api_key")
            if current_key and current_key not in ["YOUR_API_KEY_HERE", "YOUR_OPENROUTER_API_KEY_HERE"]:
                key_hint = f" [{current_key[:4]}...{current_key[-4:]}]"
            else:
                key_hint = ""

            new_key = input(f"OpenRouter API Key{key_hint} (leave blank to keep): ").strip()
            if new_key:
                if store_securely:
                    self._credential_store.store("api_key", new_key)
                    self.file_config["api_key"] = "***SECURE_STORAGE***"
                else:
                    self.file_config["api_key"] = new_key

            # 2. OpenAI API Key (optional)
            current_openai = self.get("openai_api_key")
            if current_openai:
                openai_hint = f" [{current_openai[:4]}...{current_openai[-4:]}]"
            else:
                openai_hint = ""

            new_openai = input(f"OpenAI API Key (optional){openai_hint} (leave blank to skip): ").strip()
            if new_openai:
                if store_securely:
                    self._credential_store.store("openai_api_key", new_openai)
                    self.file_config["openai_api_key"] = "***SECURE_STORAGE***"
                else:
                    self.file_config["openai_api_key"] = new_openai

            # 2b. Anthropic API Key (optional)
            current_anthropic = self.get("anthropic_api_key")
            if current_anthropic:
                anthropic_hint = f" [{current_anthropic[:4]}...{current_anthropic[-4:]}]"
            else:
                anthropic_hint = ""

            new_anthropic = input(f"Anthropic API Key (optional){anthropic_hint} (leave blank to skip): ").strip()
            if new_anthropic:
                if store_securely:
                    self._credential_store.store("anthropic_api_key", new_anthropic)
                    self.file_config["anthropic_api_key"] = "***SECURE_STORAGE***"
                else:
                    self.file_config["anthropic_api_key"] = new_anthropic

            # 3. Default Model
            current_model = self.get("model_name")
            new_model = input(f"Default Model [{current_model}]: ").strip()
            if new_model:
                self.file_config["model_name"] = new_model

            # Save
            with open(self.config_file, "w") as f:
                json.dump(self.file_config, f, indent=4)

            self.refresh()
            print("\n✅ Configuration saved to aura.config.json")
            if store_securely:
                print("🔐 API keys stored securely in OS keyring.")
            log_json("INFO", "config_interactive_bootstrap_complete")

        except EOFError:
            print("\nBootstrap cancelled.")
        except Exception as e:
            print(f"\n❌ Bootstrap failed: {e}")
            log_json("ERROR", "config_interactive_bootstrap_failed", details={"error": str(e)})

    def migrate_credentials(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Migrate plaintext API keys from config file to secure credential store.

        Security Issue #427: Migration utility for secure credential storage.

        Args:
            dry_run: If True, only report what would be migrated without making changes.

        Returns:
            Dictionary with migration results:
            {
                "migrated": [list of migrated keys],
                "already_secure": [list of keys already in secure storage],
                "not_found": [list of keys not found],
                "errors": {key: error_message},
            }
        """
        results = {
            "migrated": [],
            "already_secure": [],
            "not_found": [],
            "errors": {},
            "dry_run": dry_run,
        }

        log_json("INFO", "credential_migration_started", details={"dry_run": dry_run})

        for key in self.SECURE_CONFIG_KEYS:
            # Check if in file config
            if key not in self.file_config:
                results["not_found"].append(key)
                continue

            value = self.file_config[key]

            # Skip if already marked as secure storage
            if value == "***SECURE_STORAGE***":
                results["already_secure"].append(key)
                continue

            # Skip if empty/placeholder
            if not value or value in [
                "YOUR_API_KEY_HERE",
                "YOUR_OPENROUTER_API_KEY_HERE",
                "null",
                "None",
            ]:
                results["not_found"].append(key)
                continue

            if dry_run:
                results["migrated"].append(key)
                continue

            # Store in credential store
            try:
                if self._credential_store.store(key, value):
                    # Replace with marker in file config
                    self.file_config[key] = "***SECURE_STORAGE***"
                    results["migrated"].append(key)
                    log_json("INFO", "credential_migrated", details={"key": key})
                else:
                    results["errors"][key] = "Failed to store in credential store"
                    log_json("ERROR", "credential_migration_failed", details={"key": key})
            except Exception as e:
                results["errors"][key] = str(e)
                log_json("ERROR", "credential_migration_exception", details={"key": key, "error": str(e)})

        # Save updated config file if changes were made
        if not dry_run and results["migrated"]:
            try:
                with open(self.config_file, "w") as f:
                    json.dump(self.file_config, f, indent=4)
                log_json("INFO", "credential_migration_config_updated", details={"path": str(self.config_file)})
            except Exception as e:
                results["errors"]["_config_save"] = str(e)
                log_json("ERROR", "credential_migration_config_save_failed", details={"error": str(e)})

        # Summary
        log_json(
            "INFO",
            "credential_migration_complete",
            details={
                "migrated_count": len(results["migrated"]),
                "already_secure_count": len(results["already_secure"]),
                "error_count": len(results["errors"]),
            },
        )

        return results

    def get_credential_store_info(self) -> Dict[str, Any]:
        """Get information about the credential store configuration."""
        return self._credential_store.get_storage_info()

    def secure_store_credential(self, key: str, value: str) -> bool:
        """
        Store a credential securely in the credential store.

        Args:
            key: The credential key
            value: The credential value

        Returns:
            True if stored successfully
        """
        return self._credential_store.store(key, value)

    def secure_retrieve_credential(self, key: str) -> Optional[str]:
        """
        Retrieve a credential from the secure store.

        Args:
            key: The credential key

        Returns:
            The credential value or None
        """
        return self._credential_store.retrieve(key)

    def secure_delete_credential(self, key: str) -> bool:
        """
        Delete a credential from the secure store.

        Args:
            key: The credential key

        Returns:
            True if deleted successfully
        """
        return self._credential_store.delete(key)


# Global instance initialized with defaults
config = ConfigManager()
