# AURA Copilot onboarding

Use the repo-local setup assets before editing code with Copilot CLI:

1. Start the core AURA HTTP surfaces you need:
   - `uvicorn aura_cli.server:app --port 8001`
   - `uvicorn tools.aura_mcp_skills_server:app --port 8002`
   - optional: `uvicorn tools.sadd_mcp_server:app --port 8020`
   - optional: `uvicorn tools.github_copilot_mcp:app --port 8007`
2. Generate a local Copilot MCP config with `bash scripts/configure_copilot_mcp.sh`.
3. Confirm the repo LSP config in `.github/lsp.json` is usable with `/lsp`.
4. Use `/mcp` to verify servers are reachable before relying on them in a task.

Prefer env-based auth tokens (`AGENT_API_TOKEN`, `MCP_API_TOKEN`,
`SADD_MCP_TOKEN`, etc.). Do not store live tokens in repo files or example
configs.
