"""Typed configuration dataclasses for AURA (B5).

Provides structured, validated views over the flat config dict.
Usage:
    from core.config_manager import config
    config.llm.timeout       # int
    config.security.subprocess_timeout_s  # int
    config.loop.max_cycles   # int
    config.beads.enabled     # bool
    config.capability.auto_add  # bool
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _safe_int(val: Any, default: int) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _safe_float(val: Any, default: float) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_bool(val: Any, default: bool) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    try:
        return bool(val)
    except (ValueError, TypeError):
        return default


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider and model settings."""
    model_name: str = "google/gemini-2.0-flash-exp:free"
    api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    timeout: int = 60
    local_model_command: Optional[str] = None
    local_model_profiles: Dict[str, Any] = field(default_factory=dict)
    local_model_routing: Dict[str, Optional[str]] = field(default_factory=dict)
    model_routing: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> LLMConfig:
        return cls(
            model_name=d.get("model_name", cls.model_name),
            api_key=d.get("api_key"),
            openai_api_key=d.get("openai_api_key"),
            timeout=_safe_int(d.get("llm_timeout", cls.timeout), cls.timeout),
            local_model_command=d.get("local_model_command"),
            local_model_profiles=d.get("local_model_profiles", {}),
            local_model_routing=d.get("local_model_routing", {}),
            model_routing=d.get("model_routing", {}),
        )


@dataclass(frozen=True)
class SecurityConfig:
    """Security and sandboxing settings."""
    subprocess_timeout_s: int = 30
    max_sandbox_retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_cooldown_s: float = 60.0
    skill_timeout_s: float = 120.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SecurityConfig:
        return cls(
            subprocess_timeout_s=_safe_int(d.get("subprocess_timeout_s", cls.subprocess_timeout_s), cls.subprocess_timeout_s),
            max_sandbox_retries=_safe_int(d.get("max_sandbox_retries", cls.max_sandbox_retries), cls.max_sandbox_retries),
            circuit_breaker_threshold=_safe_int(d.get("circuit_breaker_threshold", cls.circuit_breaker_threshold), cls.circuit_breaker_threshold),
            circuit_breaker_cooldown_s=_safe_float(d.get("circuit_breaker_cooldown_s", cls.circuit_breaker_cooldown_s), cls.circuit_breaker_cooldown_s),
            skill_timeout_s=_safe_float(d.get("skill_timeout_s", cls.skill_timeout_s), cls.skill_timeout_s),
        )


@dataclass(frozen=True)
class LoopConfig:
    """Orchestration loop settings."""
    max_iterations: int = 10
    max_cycles: int = 5
    policy_name: str = "sliding_window"
    policy_max_cycles: int = 5
    policy_max_seconds: int = 120
    dry_run: bool = False
    decompose: bool = False
    strict_schema: bool = False
    evolution_trigger_every_n: int = 20
    health_monitor_trigger_every_n: int = 10
    reliability_threshold: float = 80.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> LoopConfig:
        return cls(
            max_iterations=_safe_int(d.get("max_iterations", cls.max_iterations), cls.max_iterations),
            max_cycles=_safe_int(d.get("max_cycles", cls.max_cycles), cls.max_cycles),
            policy_name=d.get("policy_name", cls.policy_name),
            policy_max_cycles=_safe_int(d.get("policy_max_cycles", cls.policy_max_cycles), cls.policy_max_cycles),
            policy_max_seconds=_safe_int(d.get("policy_max_seconds", cls.policy_max_seconds), cls.policy_max_seconds),
            dry_run=_safe_bool(d.get("dry_run", cls.dry_run), cls.dry_run),
            decompose=_safe_bool(d.get("decompose", cls.decompose), cls.decompose),
            strict_schema=_safe_bool(d.get("strict_schema", cls.strict_schema), cls.strict_schema),
            evolution_trigger_every_n=_safe_int(d.get("evolution_trigger_every_n", cls.evolution_trigger_every_n), cls.evolution_trigger_every_n),
            health_monitor_trigger_every_n=_safe_int(d.get("health_monitor_trigger_every_n", cls.health_monitor_trigger_every_n), cls.health_monitor_trigger_every_n),
            reliability_threshold=_safe_float(d.get("reliability_threshold", cls.reliability_threshold), cls.reliability_threshold),
        )


@dataclass(frozen=True)
class BeadsConfig:
    """BEADS bridge settings."""
    enabled: bool = True
    required: bool = True
    bridge_command: Optional[str] = None
    timeout_seconds: int = 20
    scope: str = "goal_run"
    persist_artifacts: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BeadsConfig:
        beads = d.get("beads", {})
        if not isinstance(beads, dict):
            beads = {}
        return cls(
            enabled=_safe_bool(beads.get("enabled", cls.enabled), cls.enabled),
            required=_safe_bool(beads.get("required", cls.required), cls.required),
            bridge_command=beads.get("bridge_command"),
            timeout_seconds=_safe_int(beads.get("timeout_seconds", cls.timeout_seconds), cls.timeout_seconds),
            scope=beads.get("scope", cls.scope),
            persist_artifacts=_safe_bool(beads.get("persist_artifacts", cls.persist_artifacts), cls.persist_artifacts),
        )


@dataclass(frozen=True)
class CapabilityConfig:
    """Capability management settings."""
    auto_add: bool = True
    auto_queue_missing: bool = True
    auto_provision_mcp: bool = False
    auto_start_mcp_servers: bool = False
    auto_backfill_coverage: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CapabilityConfig:
        return cls(
            auto_add=_safe_bool(d.get("auto_add_capabilities", cls.auto_add), cls.auto_add),
            auto_queue_missing=_safe_bool(d.get("auto_queue_missing_capabilities", cls.auto_queue_missing), cls.auto_queue_missing),
            auto_provision_mcp=_safe_bool(d.get("auto_provision_mcp", cls.auto_provision_mcp), cls.auto_provision_mcp),
            auto_start_mcp_servers=_safe_bool(d.get("auto_start_mcp_servers", cls.auto_start_mcp_servers), cls.auto_start_mcp_servers),
            auto_backfill_coverage=_safe_bool(d.get("auto_backfill_coverage", cls.auto_backfill_coverage), cls.auto_backfill_coverage),
        )


@dataclass(frozen=True)
class PathsConfig:
    """File/directory path settings."""
    memory_store_path: str = "memory/store"
    brain_db_path: str = "memory/brain_v2.db"
    goal_queue_path: str = "memory/goal_queue.json"
    goal_archive_path: str = "memory/goal_archive_v2.json"
    memory_persistence_path: str = "memory/task_hierarchy_v2.json"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PathsConfig:
        return cls(
            memory_store_path=d.get("memory_store_path", cls.memory_store_path),
            brain_db_path=d.get("brain_db_path", cls.brain_db_path),
            goal_queue_path=d.get("goal_queue_path", cls.goal_queue_path),
            goal_archive_path=d.get("goal_archive_path", cls.goal_archive_path),
            memory_persistence_path=d.get("memory_persistence_path", cls.memory_persistence_path),
        )


@dataclass(frozen=True)
class ExternalToolsConfig:
    """External CLI tool paths."""
    gemini_cli_path: str = "/data/data/com.termux/files/usr/bin/gemini"
    codex_cli_path: str = "/data/data/com.termux/files/usr/bin/codex"
    copilot_cli_path: str = "/data/data/com.termux/files/usr/bin/copilot"
    mcp_server_url: str = "http://localhost:8000"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExternalToolsConfig:
        return cls(
            gemini_cli_path=d.get("gemini_cli_path", cls.gemini_cli_path),
            codex_cli_path=d.get("codex_cli_path", cls.codex_cli_path),
            copilot_cli_path=d.get("copilot_cli_path", cls.copilot_cli_path),
            mcp_server_url=d.get("mcp_server_url", cls.mcp_server_url),
        )
