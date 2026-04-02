"""Claude Agent SDK meta-controller for AURA CLI.

Replaces rigid phase-sequenced orchestration with Claude-as-brain,
dynamically selecting workflows, tools, MCP servers, and skills
based on goal context.
"""
from core.agent_sdk.config import AgentSDKConfig

__all__ = ["AgentSDKConfig"]
