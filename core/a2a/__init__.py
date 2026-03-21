"""Agent-to-Agent (A2A) protocol support for AURA CLI.

Implements the A2A specification for agent discovery, task delegation,
and inter-agent communication. Based on the Google/Linux Foundation A2A
protocol standard.
"""
from core.a2a.agent_card import AgentCard, AgentCapability
from core.a2a.task import A2ATask, TaskState
from core.a2a.server import A2AServer
from core.a2a.client import A2AClient
