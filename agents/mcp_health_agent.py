import asyncio
from core.mcp_health import check_all_mcp_health, get_health_summary
from core.logging_utils import log_json


def _update_registry_health(results: dict) -> None:
    """Propagate health-check results to the agent registry.

    Agents whose MCP server is unhealthy are marked unavailable so that
    resolve_by_capability() skips them until they recover.
    """
    from core.mcp_agent_registry import agent_registry

    for server_name, result in results.items():
        healthy = result.get("status") == "healthy"
        # Find all agents backed by this MCP server
        for spec in agent_registry.list_agents():
            if spec.source == "mcp" and spec.mcp_server == server_name:
                if healthy:
                    agent_registry.mark_healthy(spec.name)
                else:
                    agent_registry.mark_unhealthy(spec.name)


class MCPHealthAgent:
    """
    Agent responsible for monitoring the health of all registered MCP servers.
    """

    name = "mcp_health"
    description = "Monitors and reports the health of all MCP servers."

    def run(self, input_data: dict) -> dict:
        """
        Execute the MCP health check phase.

        Args:
            input_data: Dict (currently unused).

        Returns:
            Dict containing health results and summary.
        """
        import anyio

        async def perform_check():
            results_list = await check_all_mcp_health()
            results = {r["name"]: r for r in results_list}
            _update_registry_health(results)
            summary = get_health_summary(results_list)
            return {"status": "success", "results": results_list, "summary": summary}

        try:
            try:
                asyncio.get_running_loop()
                # If we are here, a loop is running.
                # If we are in a thread, we use anyio.from_thread.run
                # If we are on the main thread, we have a problem (blocking main loop).
                # For now, assume orchestrator might be in a thread or we use from_thread.
                return anyio.from_thread.run(perform_check)
            except RuntimeError:
                # No loop running
                return anyio.run(perform_check)
        except Exception as e:
            log_json("ERROR", "mcp_health_agent_failed", details={"error": str(e)})
            return {"status": "error", "error": str(e)}
