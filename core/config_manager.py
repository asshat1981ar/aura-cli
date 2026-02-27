import os
import json
from pathlib import Path
from typing import Any, Dict, Optional
from core.logging_utils import log_json
from core.exceptions import ConfigurationError

DEFAULT_CONFIG = {
    "model_name": "google/gemini-2.0-flash-exp:free",
    "api_key": None,
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
    "model_routing": {
        "code_generation": "google/gemini-2.0-flash-exp:free",
        "planning": "google/gemini-2.0-flash-exp:free",
        "analysis": "google/gemini-2.0-flash-exp:free",
        "critique": "google/gemini-2.0-flash-exp:free",
        "embedding": "openai/text-embedding-3-small",
        "fast": "google/gemini-2.0-flash-exp:free",
        "quality": "anthropic/claude-3.5-sonnet"
    },
    # R4: MCP server port registry
    "mcp_servers": {
        "dev_tools": 8001,
        "skills": 8002,
        "control": 8003,
        "agentic_loop": 8006,
        "copilot": 8007,
    },
    # ASCM v2 Configuration
    "semantic_memory": {
        "enabled": True,
        "backend": "sqlite_local",
        "embedding_model": "text-embedding-3-small",
        "top_k": 10,
        "min_score": 0.65,
        "max_snippet_chars": 2000,
        "reindex_on_model_change": True,
        "eval_sampling_rate": 0.1
    }
}

class ConfigManager:
    """
    Unified Control Plane: Centralized configuration manager for AURA.
    Enforces a tiered strategy: (Overrides > ENV > JSON > Defaults).
    """
    def __init__(self, config_file="aura.config.json", overrides: Optional[Dict[str, Any]] = None):
        self.config_file = Path(config_file)
        self.runtime_overrides = overrides or {}
        self.file_config = {}
        self.effective_config = {}
        
        self.refresh()

    def _load_from_file(self) -> Dict[str, Any]:
        if not self.config_file.exists():
            return {}
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                log_json("INFO", "config_loaded_from_file", details={"path": str(self.config_file)})
                return data
        except json.JSONDecodeError as e:
            log_json("ERROR", "config_parse_failed", details={"error": str(e)})
            raise ConfigurationError(f"Failed to parse config file: {e}")

    def _load_from_env(self) -> Dict[str, Any]:
        env_config = {}
        for key in DEFAULT_CONFIG:
            env_key = f"AURA_{key.upper()}"
            if env_key in os.environ:
                val = os.environ[env_key]
                # Type coercion
                if isinstance(DEFAULT_CONFIG[key], bool):
                    env_config[key] = val.lower() in ("true", "1", "yes")
                elif isinstance(DEFAULT_CONFIG[key], int):
                    env_config[key] = int(val)
                else:
                    env_config[key] = val
        
        # Handle nested model_routing overrides
        routing_overrides = {}
        for sub_key in ["code_generation", "planning", "analysis", "critique", "embedding", "fast", "quality"]:
            env_name = f"AURA_MODEL_ROUTING_{sub_key.upper()}"
            if env_name in os.environ:
                routing_overrides[sub_key] = os.environ[env_name]
        if routing_overrides:
            env_config["model_routing"] = {**DEFAULT_CONFIG["model_routing"], **routing_overrides}
            
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

        return env_config

    def refresh(self):
        """Re-evaluates the effective configuration based on the tier hierarchy."""
        self.file_config = self._load_from_file()
        env_config = self._load_from_env()
        
        # Merge hierarchy: Defaults < JSON < ENV < Overrides
        merged = DEFAULT_CONFIG.copy()
        
        # Deep merge for model_routing and semantic_memory to preserve defaults if partial override
        if "model_routing" in self.file_config:
            merged["model_routing"].update(self.file_config["model_routing"])
        if "semantic_memory" in self.file_config:
            merged["semantic_memory"].update(self.file_config["semantic_memory"])
            
        # Update other top-level keys
        for k, v in self.file_config.items():
            if k not in ["model_routing", "semantic_memory"]:
                merged[k] = v
        
        # Merge ENV (env_config already has merged sub-dicts)
        # But we need to be careful not to overwrite file_config partials with defaults if env is empty
        # Actually _load_from_env returns only set keys + merged sub-dicts if env vars exist.
        
        if "model_routing" in env_config:
            merged["model_routing"].update(env_config["model_routing"])
        if "semantic_memory" in env_config:
            merged["semantic_memory"].update(env_config["semantic_memory"])
            
        for k, v in env_config.items():
            if k not in ["model_routing", "semantic_memory"]:
                merged[k] = v
                
        merged.update(self.runtime_overrides)
        
        self.effective_config = merged

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a configuration value from the effective config."""
        return self.effective_config.get(key, default)

    def set_runtime_override(self, key: str, value: Any):
        """Sets a temporary runtime override."""
        self.runtime_overrides[key] = value
        self.refresh()

    def persist_to_file(self, key: str, value: Any):
        """Sets a configuration value and persists it to the aura.config.json file."""
        self.file_config[key] = value
        try:
            with open(self.config_file, 'w') as f:
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
                raise ConfigurationError(
                    f"Unknown MCP server name '{server_name}'. "
                    f"Known servers: {list(default_registry.keys())}"
                )
            return int(default_registry[server_name])
        return int(registry[server_name])

    def bootstrap(self):
        """Generates a default aura.config.json if it doesn't exist."""
        if self.config_file.exists():
            log_json("INFO", "config_bootstrap_skipped_exists")
            return
        
        # Create minimal valid config
        bootstrap_data = {
            "model_name": DEFAULT_CONFIG["model_name"],
            "api_key": "YOUR_OPENROUTER_API_KEY_HERE"
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(bootstrap_data, f, indent=4)
        log_json("INFO", "config_bootstrapped", details={"path": str(self.config_file)})

# Global instance initialized with defaults
config = ConfigManager()
