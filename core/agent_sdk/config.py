"""Configuration for the Agent SDK meta-controller."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Default MCP server ports matching aura.config.json
_DEFAULT_MCP_PORTS: Dict[str, int] = {
    "dev_tools": 8001,
    "skills": 8002,
    "control": 8003,
    "thinking": 8004,
    "agentic_loop": 8006,
    "copilot": 8007,
    "sadd": 8020,
    "discovery": 8025,
}

# Built-in tools from Agent SDK that we always allow
_DEFAULT_ALLOWED_TOOLS: List[str] = [
    "Read", "Write", "Edit", "Bash", "Glob", "Grep", "Agent",
]


@dataclass
class AgentSDKConfig:
    """Configuration for Agent SDK meta-controller sessions."""

    model: str = "claude-sonnet-4-6"
    max_turns: int = 30
    max_budget_usd: float = 2.0
    permission_mode: str = "acceptEdits"
    allowed_tools: List[str] = field(default_factory=lambda: list(_DEFAULT_ALLOWED_TOOLS))
    mcp_ports: Dict[str, int] = field(default_factory=lambda: dict(_DEFAULT_MCP_PORTS))
    project_root: Optional[str] = None
    enable_thinking: bool = True
    enable_subagents: bool = True
    enable_hooks: bool = True
    # Model router config
    model_stats_path: Path = field(default_factory=lambda: Path("memory/agent_sdk_model_stats.json"))
    escalation_threshold: int = 2
    de_escalation_threshold: int = 5
    min_success_rate: float = 0.7
    ema_alpha: float = 0.2
    # Session persistence
    session_db_path: Path = field(default_factory=lambda: Path("memory/agent_sdk_sessions.db"))
    # Skill weight updater
    skill_weights_path: Path = field(default_factory=lambda: Path("memory/skill_weights.json"))
    skill_weight_success_delta: float = 0.1
    skill_weight_failure_delta: float = -0.05
    skill_weight_cap: float = 1.0
    skill_weight_floor: float = 0.1

    @classmethod
    def from_aura_config(cls, aura_config: Dict[str, Any]) -> "AgentSDKConfig":
        """Build config from aura.config.json dict."""
        sdk_section = aura_config.get("agent_sdk", {})
        mcp_ports = aura_config.get("mcp_servers", dict(_DEFAULT_MCP_PORTS))
        return cls(
            model=sdk_section.get("model", cls.model),
            max_turns=sdk_section.get("max_turns", cls.max_turns),
            max_budget_usd=sdk_section.get("max_budget_usd", cls.max_budget_usd),
            permission_mode=sdk_section.get("permission_mode", cls.permission_mode),
            allowed_tools=sdk_section.get("allowed_tools", list(_DEFAULT_ALLOWED_TOOLS)),
            mcp_ports=mcp_ports,
            project_root=sdk_section.get("project_root"),
            enable_thinking=sdk_section.get("enable_thinking", True),
            enable_subagents=sdk_section.get("enable_subagents", True),
            enable_hooks=sdk_section.get("enable_hooks", True),
            model_stats_path=Path(sdk_section.get("model_stats_path", "memory/agent_sdk_model_stats.json")),
            escalation_threshold=sdk_section.get("escalation_threshold", 2),
            de_escalation_threshold=sdk_section.get("de_escalation_threshold", 5),
            min_success_rate=sdk_section.get("min_success_rate", 0.7),
            ema_alpha=sdk_section.get("ema_alpha", 0.2),
            session_db_path=Path(sdk_section.get("session_db_path", "memory/agent_sdk_sessions.db")),
            skill_weights_path=Path(sdk_section.get("skill_weights_path", "memory/skill_weights.json")),
            skill_weight_success_delta=sdk_section.get("skill_weight_success_delta", 0.1),
            skill_weight_failure_delta=sdk_section.get("skill_weight_failure_delta", -0.05),
            skill_weight_cap=sdk_section.get("skill_weight_cap", 1.0),
            skill_weight_floor=sdk_section.get("skill_weight_floor", 0.1),
        )

    def apply_env_overrides(self) -> None:
        """Override config values from environment variables."""
        if model := os.environ.get("AURA_AGENT_SDK_MODEL"):
            self.model = model
        if max_turns := os.environ.get("AURA_AGENT_SDK_MAX_TURNS"):
            self.max_turns = int(max_turns)
        if budget := os.environ.get("AURA_AGENT_SDK_MAX_BUDGET"):
            self.max_budget_usd = float(budget)
        if mode := os.environ.get("AURA_AGENT_SDK_PERMISSION_MODE"):
            self.permission_mode = mode

    @property
    def mcp_server_endpoints(self) -> Dict[str, str]:
        """Return {name: url} for all configured MCP servers."""
        return {
            name: f"http://localhost:{port}"
            for name, port in self.mcp_ports.items()
        }

    @property
    def thinking_config(self) -> Optional[Dict[str, str]]:
        """Return thinking config for Agent SDK options."""
        if self.enable_thinking:
            return {"type": "adaptive"}
        return None
