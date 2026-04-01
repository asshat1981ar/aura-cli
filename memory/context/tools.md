# AURA Tools & Environment

## Development Tools

| Tool | Purpose | Location |
|------|---------|----------|
| Python 3.13 | Runtime | System |
| uv | Python package manager | `~/.local/share/uv/` |
| npx | Node execution | System |
| n8n | Workflow automation | http://localhost:5678 |
| Smithery | MCP server management | CLI |

## AURA CLI Commands

```bash
# Goal management
aura goal status          # Show goal queue status
aura goal add "..."       # Add new goal
aura goal run             # Execute next goal

# SADD execution
aura sadd spec.md         # Run SADD with spec

# Agent Swarm
export AURA_ENABLE_SWARM=1
aura swarm status         # Check swarm health
```

## n8n Workflows

| Workflow | Purpose | Status | Webhook |
|----------|---------|--------|---------|
| WF-0 | Master dispatcher | 🟢 **ACTIVE** | `/fleet` (POST) |
| WF-1 | Bug fix handler | 🟢 **ACTIVE** | Internal trigger |
| WF-2 | Feature handler | 🟢 **ACTIVE** | Internal trigger |
| WF-3 | Refactor handler | 🟢 **ACTIVE** | Internal trigger |
| WF-4 | Security handler | 🟢 **ACTIVE** | Internal trigger |
| WF-5 | Docs handler | 🟢 **ACTIVE** | Internal trigger |
| WF-6 | Code gen & PR | 🟢 **ACTIVE** | Internal trigger |

**Fleet Dispatcher Entry Point:**
```bash
# Trigger workflow fleet via webhook
curl -X POST http://localhost:5678/webhook/fleet \
  -H "Content-Type: application/json" \
  -d '{
    "action": "opened",
    "issue": {"number": 123, "title": "Bug: ..."},
    "repository": {"full_name": "owner/repo"}
  }'
```

**n8n Console:** http://localhost:5678 (Owner setup required for first access)

**Workflow Files:** `n8n-workflows/WF-*.json`

## MCP Servers

| Server | Type | Auth Status |
|--------|------|-------------|
| github | stdio | ✅ Connected |
| slack | stdio | ⏳ Pending |
| sentry | stdio | ⏳ Pending |
| supabase | stdio | ⏳ Pending |

## Environment Variables

```bash
AURA_ENABLE_SWARM=1       # Enable Agent Swarm
AURA_DB_PATH=...          # SQLite database path
AURA_N8N_URL=...          # n8n webhook URL
OPENROUTER_API_KEY=...    # LLM routing
```

## Key Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Hot cache memory |
| `plans/sadd-execution-spec.md` | SADD specifications |
| `~/.config/agents/skills/` | Installed skills |
| `scripts/n8n-mcp-proxy.mjs` | MCP proxy for n8n |
