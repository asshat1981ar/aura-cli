# AURA MCP usage

This repository exposes multiple MCP-compatible HTTP servers for Copilot workflows:

- `aura-dev-tools` on port `8001` for runtime tools and goal execution
- `aura-skills` on port `8002` for registered skill tools
- `aura-control` on port `8003` for the control-plane API
- `aura-agentic-loop` on port `8006` for loop orchestration
- `aura-copilot` on port `8007` for GitHub/Copilot-specific workflows

When documenting or editing MCP setup:

- prefer `.vscode/mcp.json.example` and `scripts/configure_copilot_mcp.sh`
- keep auth token values in environment variables
- keep ports aligned with `aura.config.json` / `settings.json`
- avoid hardcoded skill counts when the registry is the real source of truth
