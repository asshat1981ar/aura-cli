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
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    mcp_servers = config.get("mcpServers", {})
                    for name, details in mcp_servers.items():
                        # Basic validation of config format
                        if "command" in details:
                            discovered.append({
                                "name": name,
                                "command": details["command"],
                                "status": "configured"
                            })
                        else:
                            log_json("WARNING", "mcp_discovery_invalid_config", details={"server": name})
            except json.JSONDecodeError as e:
                log_json("ERROR", "mcp_discovery_json_error", details={"error": str(e)})
                return {"status": "error", "error": f"Invalid {self.config_path} format", "discovered": []}
        else:
            return {"status": "success", "message": f"No {self.config_path} found.", "discovered": []}
                
        return {
            "status": "success",
            "discovered": discovered,
            "message": f"Found {len(discovered)} MCP servers in config."
        }
