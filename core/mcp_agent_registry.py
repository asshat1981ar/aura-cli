from __future__ import annotations
from typing import Dict, List, Optional
from core.types import AgentSpec, MCPServerConfig
from core.logging_utils import log_json

class TypedAgentRegistry:
    """
    A modern, typed registry for managing both local and MCP-backed agents.
    Supports resolution by capability and deterministic precedence.
    """
    def __init__(self):
        self._agents: Dict[str, AgentSpec] = {}
        self._capabilities: Dict[str, List[str]] = {} # capability -> list of agent names

    def register(self, spec: AgentSpec, overwrite: bool = False):
        """Register an agent specification."""
        if spec.name in self._agents and not overwrite:
            log_json("ERROR", "agent_registration_conflict", details={"name": spec.name})
            raise ValueError(f"Agent '{spec.name}' is already registered.")
        
        self._agents[spec.name] = spec
        
        # Index capabilities
        for cap in spec.capabilities:
            if cap not in self._capabilities:
                self._capabilities[cap] = []
            if spec.name not in self._capabilities[cap]:
                self._capabilities[cap].append(spec.name)
        
        log_json("INFO", "agent_registered", details={
            "name": spec.name,
            "source": spec.source,
            "capabilities": spec.capabilities
        })

    def get_agent(self, name: str) -> Optional[AgentSpec]:
        """Retrieve an agent by name."""
        return self._agents.get(name)

    def resolve_by_capability(self, capability: str) -> List[AgentSpec]:
        """
        Find agents that support a specific capability.
        Precedence: local agents first, then MCP agents.
        """
        names = self._capabilities.get(capability, [])
        results = [self._agents[name] for name in names]
        
        # Sort: local agents first
        results.sort(key=lambda x: 0 if x.source == "local" else 1)
        return results

    def list_agents(self) -> List[AgentSpec]:
        """Return all registered agent specifications."""
        return list(self._agents.values())

    async def register_mcp_agents(self, server_config: MCPServerConfig):
        """Discover and register agents from an MCP server."""
        from core.mcp_client import MCPAsyncClient
        
        url = f"http://127.0.0.1:{server_config.port}"
        client = MCPAsyncClient(url)
        try:
            tools = await client.get_tools()
            for tool in tools:
                spec = AgentSpec(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    capabilities=[tool["name"]], # Simplistic mapping for now
                    source="mcp",
                    mcp_server=server_config.name
                )
                self.register(spec, overwrite=True)
        except Exception as e:
            log_json("ERROR", "mcp_agent_discovery_failed", details={
                "server": server_config.name,
                "error": str(e)
            })

    def clear(self):
        """Clear the registry (mostly for testing)."""
        self._agents.clear()
        self._capabilities.clear()

# Global singleton for the typed registry
agent_registry = TypedAgentRegistry()
