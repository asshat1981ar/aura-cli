"""Dynamic agent and server registry for the orchestrator hub.

Extends the concept from agents/registry.py to a multi-environment context
with dynamic discovery and capability-based lookup.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AgentInfo:
    """Registered agent metadata."""

    name: str
    agent_type: str
    capabilities: List[str]
    endpoint: str
    environment: str
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    status: str = "active"
    metadata: Dict = field(default_factory=dict)

    def as_dict(self) -> Dict:
        return {
            "name": self.name,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "endpoint": self.endpoint,
            "environment": self.environment,
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
            "status": self.status,
            "metadata": self.metadata,
        }


@dataclass
class ServerInfo:
    """Registered MCP server metadata."""

    name: str
    port: int
    server_type: str  # "internal_http", "external_stdio"
    status: str = "unknown"
    health_path: str = "/health"
    last_check: float = 0.0
    tools: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {
            "name": self.name,
            "port": self.port,
            "server_type": self.server_type,
            "status": self.status,
            "health_path": self.health_path,
            "last_check": self.last_check,
            "tools": self.tools,
        }


class AgentRegistryHub:
    """Multi-environment agent and server registry with dynamic discovery."""

    def __init__(self):
        self._agents: Dict[str, AgentInfo] = {}
        self._servers: Dict[str, ServerInfo] = {}

    # -- Agent operations --

    def register_agent(
        self,
        name: str,
        agent_type: str,
        capabilities: List[str],
        endpoint: str,
        environment: str,
        metadata: Optional[Dict] = None,
    ) -> AgentInfo:
        """Register or update an agent."""
        info = AgentInfo(
            name=name,
            agent_type=agent_type,
            capabilities=capabilities,
            endpoint=endpoint,
            environment=environment,
            metadata=metadata or {},
        )
        self._agents[name] = info
        return info

    def deregister_agent(self, name: str) -> bool:
        """Remove an agent from the registry."""
        return self._agents.pop(name, None) is not None

    def discover(self, capability: str) -> List[AgentInfo]:
        """Find all agents that provide a given capability."""
        return [
            agent
            for agent in self._agents.values()
            if capability in agent.capabilities and agent.status == "active"
        ]

    def discover_by_type(self, agent_type: str) -> List[AgentInfo]:
        """Find all agents of a given type."""
        return [
            agent
            for agent in self._agents.values()
            if agent.agent_type == agent_type and agent.status == "active"
        ]

    def discover_by_environment(self, environment: str) -> List[AgentInfo]:
        """Find all agents in a given environment."""
        return [
            agent
            for agent in self._agents.values()
            if agent.environment == environment
        ]

    def heartbeat(self, name: str) -> bool:
        """Update agent heartbeat timestamp."""
        agent = self._agents.get(name)
        if agent is None:
            return False
        agent.last_heartbeat = time.time()
        agent.status = "active"
        return True

    def list_agents(self) -> List[AgentInfo]:
        """List all registered agents."""
        return list(self._agents.values())

    def get_agent(self, name: str) -> Optional[AgentInfo]:
        """Get agent info by name."""
        return self._agents.get(name)

    def mark_unhealthy(self, name: str) -> None:
        """Mark an agent as unhealthy."""
        agent = self._agents.get(name)
        if agent:
            agent.status = "unhealthy"

    # -- Server operations --

    def register_server(
        self,
        name: str,
        port: int,
        server_type: str = "internal_http",
        health_path: str = "/health",
        tools: Optional[List[str]] = None,
    ) -> ServerInfo:
        """Register or update an MCP server."""
        info = ServerInfo(
            name=name,
            port=port,
            server_type=server_type,
            health_path=health_path,
            tools=tools or [],
        )
        self._servers[name] = info
        return info

    def deregister_server(self, name: str) -> bool:
        """Remove a server from the registry."""
        return self._servers.pop(name, None) is not None

    def list_servers(self) -> List[ServerInfo]:
        """List all registered servers."""
        return list(self._servers.values())

    def get_server(self, name: str) -> Optional[ServerInfo]:
        """Get server info by name."""
        return self._servers.get(name)

    def update_server_status(self, name: str, status: str) -> None:
        """Update server health status."""
        server = self._servers.get(name)
        if server:
            server.status = status
            server.last_check = time.time()
