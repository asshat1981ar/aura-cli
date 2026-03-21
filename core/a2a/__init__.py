"""Agent-to-Agent (A2A) protocol support for AURA CLI.

Implements the A2A specification for agent discovery, task delegation,
and inter-agent communication. Based on the Google/Linux Foundation A2A
protocol standard.
"""
from core.a2a.agent_card import AgentCard as AgentCard, AgentCapability as AgentCapability
from core.a2a.task import A2ATask as A2ATask, TaskState as TaskState
from core.a2a.server import A2AServer as A2AServer
from core.a2a.client import A2AClient as A2AClient
