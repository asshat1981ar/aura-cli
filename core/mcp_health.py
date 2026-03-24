import asyncio
from typing import Dict, List, Any
from core.mcp_client import MCPAsyncClient
from core.config_manager import config
from core.logging_utils import log_json

async def check_mcp_health(server_name: str) -> Dict[str, Any]:
    """Check the health of a specific named MCP server."""
    try:
        port = config.get_mcp_server_port(server_name)
        url = f"http://127.0.0.1:{port}"
        client = MCPAsyncClient(url, timeout=5)
        health = await client.get_health()
        return {
            "name": server_name,
            "port": port,
            "status": "healthy",
            "health_data": health
        }
    except Exception as e:
        log_json("WARN", "mcp_health_check_failed", details={"server": server_name, "error": str(e)})
        return {
            "name": server_name,
            "status": "unhealthy",
            "error": str(e)
        }

async def check_all_mcp_health() -> List[Dict[str, Any]]:
    """Check the health of all registered MCP servers (ports 8001-8007)."""
    mcp_config = config.get("mcp_servers", {})
    tasks = [check_mcp_health(name) for name in mcp_config.keys()]
    return await asyncio.gather(*tasks)

def get_health_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a summary of health check results."""
    total = len(results)
    healthy = sum(1 for r in results if r["status"] == "healthy")
    return {
        "total_servers": total,
        "healthy_count": healthy,
        "unhealthy_count": total - healthy,
        "all_healthy": healthy == total
    }
