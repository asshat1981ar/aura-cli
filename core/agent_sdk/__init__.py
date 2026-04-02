"""Claude Agent SDK meta-controller for AURA CLI.

Replaces rigid phase-sequenced orchestration with Claude-as-brain,
dynamically selecting workflows, tools, MCP servers, and skills
based on goal context.
"""
from core.agent_sdk.config import AgentSDKConfig
from core.agent_sdk.controller import AuraController
from core.agent_sdk.cli_integration import build_controller_from_args, handle_agent_run

__all__ = [
    "AgentSDKConfig",
    "AuraController",
    "build_controller_from_args",
    "handle_agent_run",
]
