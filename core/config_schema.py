"""Pydantic schema for AURA configuration validation.

Provides type-safe configuration with validation at startup,
preventing AttributeError deep in the call stack.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Literal

pydantic_available = False
try:
    from pydantic import BaseModel, Field, field_validator, ValidationError
    pydantic_available = True
except ImportError:
    # Fallback when Pydantic is not installed
    BaseModel = object
    Field = lambda **kwargs: None
    field_validator = lambda *args, **kwargs: lambda f: f
    ValidationError = Exception


class SemanticMemoryConfig(BaseModel if pydantic_available else object):
    """Configuration for semantic memory (ASCM v2)."""
    enabled: bool = True
    backend: Literal["sqlite_local", "qdrant"] = "sqlite_local"
    embedding_model: str = "text-embedding-3-small"
    top_k: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.65, ge=0.0, le=1.0)
    max_snippet_chars: int = Field(default=2000, ge=100, le=10000)
    reindex_on_model_change: bool = True
    eval_sampling_rate: float = Field(default=0.1, ge=0.0, le=1.0)


class ModelRoutingConfig(BaseModel if pydantic_available else object):
    """Model routing configuration for different task types."""
    code_generation: str = "google/gemini-2.0-flash-exp:free"
    planning: str = "google/gemini-2.0-flash-exp:free"
    analysis: str = "google/gemini-2.0-flash-exp:free"
    critique: str = "google/gemini-2.0-flash-exp:free"
    embedding: str = "openai/text-embedding-3-small"
    fast: str = "google/gemini-2.0-flash-exp:free"
    quality: str = "anthropic/claude-3.5-sonnet"


class BeadsConfig(BaseModel if pydantic_available else object):
    """BEADS (Behavioral Evolution and Design System) configuration."""
    enabled: bool = True
    required: bool = True
    bridge_command: Optional[str] = None
    timeout_seconds: int = Field(default=20, ge=1, le=300)
    scope: Literal["goal_run", "session", "project"] = "goal_run"
    persist_artifacts: bool = True


class McpServersConfig(BaseModel if pydantic_available else object):
    """MCP server port registry."""
    dev_tools: int = Field(default=8001, ge=1024, le=65535)
    skills: int = Field(default=8002, ge=1024, le=65535)
    control: int = Field(default=8003, ge=1024, le=65535)
    agentic_loop: int = Field(default=8006, ge=1024, le=65535)
    copilot: int = Field(default=8007, ge=1024, le=65535)


class AuraConfig(BaseModel if pydantic_available else object):
    """Main AURA configuration schema with validation.
    
    This Pydantic model validates configuration at startup,
    providing clear error messages for invalid settings.
    """
    
    # Model settings
    model_name: str = "google/gemini-2.0-flash-exp:free"
    api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Execution settings
    dry_run: bool = False
    decompose: bool = False
    max_iterations: int = Field(default=10, ge=1, le=1000)
    max_cycles: int = Field(default=5, ge=1, le=100)
    strict_schema: bool = False
    
    # Policy settings
    policy_name: Literal["sliding_window", "fixed", "unlimited"] = "sliding_window"
    policy_max_cycles: int = Field(default=5, ge=1, le=100)
    policy_max_seconds: int = Field(default=120, ge=1, le=3600)
    
    # Path settings
    memory_persistence_path: str = "memory/task_hierarchy_v2.json"
    memory_store_path: str = "memory/store"
    goal_queue_path: str = "memory/goal_queue.json"
    goal_archive_path: str = "memory/goal_archive_v2.json"
    brain_db_path: str = "memory/brain_v2.db"
    
    # Automation settings
    auto_add_capabilities: bool = True
    auto_queue_missing_capabilities: bool = True
    auto_provision_mcp: bool = False
    auto_start_mcp_servers: bool = False
    auto_backfill_coverage: bool = True
    
    # Orchestrator settings
    enable_mcp_registry: bool = True
    enable_new_orchestrator: bool = True
    force_legacy_orchestrator: bool = False
    new_orchestrator_shadow_mode: bool = False
    reliability_threshold: float = Field(default=80.0, ge=0.0, le=100.0)
    
    # CLI paths
    gemini_cli_path: Optional[str] = "/data/data/com.termux/files/usr/bin/gemini"
    codex_cli_path: Optional[str] = "/data/data/com.termux/files/usr/bin/codex"
    copilot_cli_path: Optional[str] = "/data/data/com.termux/files/usr/bin/copilot"
    
    # MCP settings
    mcp_server_url: str = "http://localhost:8000"
    local_model_command: Optional[str] = None
    local_model_profiles: Dict[str, Any] = Field(default_factory=dict)
    local_model_routing: Dict[str, Optional[str]] = Field(default_factory=lambda: {
        "planning": None,
        "analysis": None,
        "critique": None,
        "code_generation": None,
        "embedding": None,
        "quality": None,
        "fast": None,
    })
    llm_timeout: int = Field(default=60, ge=1, le=600)
    
    # Nested configurations
    beads: BeadsConfig = Field(default_factory=BeadsConfig)
    model_routing: ModelRoutingConfig = Field(default_factory=ModelRoutingConfig)
    mcp_servers: McpServersConfig = Field(default_factory=McpServersConfig)
    semantic_memory: SemanticMemoryConfig = Field(default_factory=SemanticMemoryConfig)
    
    @field_validator("memory_persistence_path", "memory_store_path", 
                     "goal_queue_path", "goal_archive_path", "brain_db_path")
    @classmethod
    def validate_paths(cls, v: str) -> str:
        """Ensure paths are valid strings."""
        if not v or not isinstance(v, str):
            raise ValueError("Path must be a non-empty string")
        return v


class ConfigValidator:
    """Configuration validator with Pydantic support.
    
    Falls back to legacy validation when Pydantic is not available.
    """
    
    def __init__(self):
        self.use_pydantic = pydantic_available
    
    def validate(self, config_dict: Dict[str, Any]) -> tuple[bool, Dict[str, Any], list[str]]:
        """Validate configuration dictionary.
        
        Args:
            config_dict: Configuration to validate
            
        Returns:
            Tuple of (is_valid, validated_config, error_messages)
        """
        if not self.use_pydantic:
            # Fallback: return config as-is with basic type checking
            errors = self._legacy_validate(config_dict)
            return len(errors) == 0, config_dict, errors
        
        try:
            # Use Pydantic for validation
            config = AuraConfig(**config_dict)
            return True, config.model_dump(), []
        except ValidationError as e:
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            return False, config_dict, errors
        except Exception as e:
            return False, config_dict, [str(e)]
    
    def _legacy_validate(self, config_dict: Dict[str, Any]) -> list[str]:
        """Legacy validation when Pydantic is unavailable."""
        errors = []
        
        # Basic type validation
        int_fields = ["max_iterations", "max_cycles", "policy_max_cycles", 
                      "policy_max_seconds", "llm_timeout"]
        for field in int_fields:
            if field in config_dict and not isinstance(config_dict[field], int):
                errors.append(f"{field} must be an integer")
        
        bool_fields = ["dry_run", "decompose", "strict_schema", "auto_add_capabilities"]
        for field in bool_fields:
            if field in config_dict and not isinstance(config_dict[field], bool):
                errors.append(f"{field} must be a boolean")
        
        return errors
    
    def get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values."""
        if self.use_pydantic:
            return AuraConfig().model_dump()
        return {}


def validate_config(config_dict: Dict[str, Any]) -> tuple[bool, list[str]]:
    """Convenience function to validate configuration.
    
    Args:
        config_dict: Configuration to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    validator = ConfigValidator()
    is_valid, _, errors = validator.validate(config_dict)
    return is_valid, errors


# Feature flag check
def is_pydantic_available() -> bool:
    """Check if Pydantic is available for config validation."""
    return pydantic_available
