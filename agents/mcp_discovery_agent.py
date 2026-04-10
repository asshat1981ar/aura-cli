import json
from pathlib import Path
from core.logging_utils import log_json


class MCPDiscoveryAgent:
    """
    Agent responsible for dynamically discovering, validating, and provisioning
    new MCP (Model Context Protocol) servers in the current workspace.
    """

    name = "mcp_discovery"

    def __init__(self, config_path=".mcp.json"):
        """Initialise the MCP Discovery Agent."""
        self.config_path = config_path

    def run(self, input_data: dict) -> dict:
        """
        Execute the MCP discovery phase.

        Args:
            input_data: Dict containing optional 'project_root' or 'goal'.

        Returns:
            Dict containing discovered servers and their status.
        """
        project_root = Path(input_data.get("project_root", Path.cwd()))
        config_file = project_root / self.config_path

        discovered = []
        mcp_servers = {}
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    mcp_servers = config.get("mcpServers", {})
                    for name, details in mcp_servers.items():
                        # Basic validation of config format — support both stdio (command) and HTTP (url) transports
                        if "command" in details:
                            discovered.append({"name": name, "command": details["command"], "transport": "stdio", "status": "configured"})
                        elif "url" in details:
                            discovered.append({"name": name, "url": details["url"], "transport": "http", "status": "configured"})
                        else:
                            log_json("WARNING", "mcp_discovery_invalid_config", details={"server": name})
            except json.JSONDecodeError as e:
                log_json("ERROR", "mcp_discovery_json_error", details={"error": str(e)})
                return {"status": "error", "error": f"Invalid {self.config_path} format", "discovered": []}
        else:
            return {"status": "success", "message": f"No {self.config_path} found.", "discovered": []}

        self._register_mcp_agents_for_discovered(discovered, mcp_servers)
        return {"status": "success", "discovered": discovered, "message": f"Found {len(discovered)} MCP servers in config."}

    def _register_mcp_agents_for_discovered(self, discovered: list, mcp_servers: dict) -> None:
        """Attempt to register MCP agents for each discovered HTTP server."""
        from core.mcp_agent_registry import agent_registry
        from core.types import MCPServerConfig
        import asyncio
        import anyio

        async def _do_register():
            for server in discovered:
                server_raw = mcp_servers.get(server.get("name", ""), {})
                port = server_raw.get("port") or (
                    int(server_raw.get("env", {}).get("PORT", 0)) or None
                )
                if port:
                    cfg = MCPServerConfig(
                        name=server["name"],
                        command=server.get("command", ""),
                        port=port,
                    )
                    try:
                        await agent_registry.register_mcp_agents(cfg)
                    except Exception as e:
                        log_json("WARN", "mcp_discovery_register_failed",
                                 details={"server": server["name"], "error": str(e)})

        try:
            asyncio.get_running_loop()
            anyio.from_thread.run(_do_register)
        except RuntimeError:
            anyio.run(_do_register)
