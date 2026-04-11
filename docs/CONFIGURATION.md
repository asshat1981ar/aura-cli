# AURA CLI — Configuration Reference

AURA resolves configuration from four layers, applied in descending priority:

```
Environment Variables  (highest)
        ↓
  aura.config.json     (project-level config file)
        ↓
   settings.json       (legacy / editor-integration config)
        ↓
     Defaults          (built-in Pydantic schema defaults)  (lowest)
```

A value set at a higher layer always wins over a lower one.

---

## 1 · Config Precedence

### Layer 1 — Environment Variables (highest priority)

Set in your shell, a `.env` file (loaded via `python-dotenv`), or CI secrets.
These override every other source.

```bash
# Override the primary model for every run:
export OPENROUTER_API_KEY=sk-or-...
export AURA_MODEL_ROUTING_CODE_GENERATION=anthropic/claude-3.5-sonnet

# Run a single goal in dry-run mode:
AURA_DRY_RUN=true aura run "refactor utils.py"
```

### Layer 2 — aura.config.json (project-level)

Place `aura.config.json` in your project root (auto-discovered) or point to it
explicitly with `AURA_CONFIG_PATH`.

```jsonc
// aura.config.json
{
  "model_name": "google/gemini-2.0-flash-exp:free",
  "max_iterations": 15,
  "dry_run": false,
  "model_routing": {
    "code_generation": "anthropic/claude-3.5-sonnet",
    "planning": "deepseek/deepseek-chat",
    "analysis": "google/gemini-2.5-flash",
    "critique": "deepseek/deepseek-r1-0528",
    "embedding": "openai/text-embedding-3-small",
    "fast": "google/gemini-2.0-flash-001",
    "quality": "deepseek/deepseek-r1-0528"
  },
  "beads": {
    "enabled": true,
    "scope": "goal_run"
  },
  "semantic_memory": {
    "enabled": true,
    "backend": "sqlite_local",
    "top_k": 10
  }
}
```

### Layer 3 — settings.json (legacy / editor integration)

`settings.json` in the project root is loaded as a fallback when
`aura.config.json` is absent.  Supports the same fields.  Preferred by the
VS Code extension.

```jsonc
// settings.json
{
  "model_name": "google/gemini-2.0-flash-exp:free",
  "max_iterations": 10
}
```

### Layer 4 — Defaults (lowest priority)

Built-in defaults are defined in `core/config_schema.py` as Pydantic field
defaults.  Run the following to see the full default set:

```bash
python3 -c "from core.config_schema import AuraConfig; import json; print(json.dumps(AuraConfig().model_dump(), indent=2))"
```

---

## 2 · Environment Variables

All variables are optional unless marked **[required]** or **[secret]**.

### Core Runtime

| Variable | Default | Description |
|---|---|---|
| `AURA_ENV` | `development` | Deployment environment: `development`, `staging`, `production` |
| `AURA_LOG_LEVEL` | `info` | Log verbosity: `debug`, `info`, `warn`, `error` |
| `AURA_SKIP_CHDIR` | unset | Set to `1` to prevent the CLI from `chdir`-ing to the project root (needed in CI/pytest) |
| `AURA_DRY_RUN` | `false` | Simulate goal execution without LLM calls or file writes |
| `AURA_PROJECT_ROOT` | cwd | Override project root detected by the server |
| `AURA_CONFIG_PATH` | auto | Explicit path to `aura.config.json` / `settings.json` |
| `AURA_TEST_MODE` | unset | Set to `1` in tests to suppress side-effects |
| `AURA_RUN_TOOL_TIMEOUT_S` | `15` | Max seconds a `/execute?tool=run` subprocess may run (1–60) |
| `AURA_RUN_TOOL_MAX_OUTPUT_BYTES` | `65536` | Max bytes captured from subprocess output |

### API Server

| Variable | Default | Description |
|---|---|---|
| `AURA_API_HOST` | `0.0.0.0` | Host the FastAPI server binds to |
| `AURA_API_PORT` | `8000` | Port the FastAPI server listens on |
| `AURA_API_WORKERS` | `1` | Number of uvicorn worker processes |
| `AGENT_API_TOKEN` | unset | **[secret]** Bearer token for all authenticated API endpoints |
| `AGENT_API_ENABLE_RUN` | `0` | Set to `1` to enable the shell-execution endpoint |

### Model Routing

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | **[secret] [required]** OpenRouter API key for cloud model access |
| `ANTHROPIC_API_KEY` | — | **[secret]** Direct Anthropic API key |
| `OPENAI_API_KEY` | — | **[secret]** OpenAI API key (also for embeddings) |
| `AURA_LLM_TIMEOUT` | `60` | LLM call timeout in seconds (1–600) |
| `AURA_MODEL_ROUTING_CODE_GENERATION` | see schema | Override model for code-generation tasks |
| `AURA_MODEL_ROUTING_PLANNING` | see schema | Override model for planning tasks |
| `AURA_MODEL_ROUTING_ANALYSIS` | see schema | Override model for analysis tasks |
| `AURA_MODEL_ROUTING_CRITIQUE` | see schema | Override model for critique tasks |
| `AURA_MODEL_ROUTING_EMBEDDING` | see schema | Override embedding model |
| `AURA_MODEL_ROUTING_FAST` | see schema | Override model for low-latency tasks |
| `AURA_MODEL_ROUTING_QUALITY` | see schema | Override model for highest-quality tasks |
| `AURA_LOCAL_MODEL_COMMAND` | unset | Command to invoke a local LLM (e.g. llama-server) |

### MCP Auth Tokens

| Variable | Port | Description |
|---|---|---|
| `AGENT_API_TOKEN` | 8001 | **[secret]** aura-dev-tools MCP server |
| `MCP_API_TOKEN` | 8002 | **[secret]** aura-skills MCP server |
| `MCP_CONTROL_TOKEN` | 8003 | **[secret]** aura-control MCP server |
| `AGENTIC_LOOP_TOKEN` | 8006 | **[secret]** aura-agentic-loop MCP server |
| `COPILOT_MCP_TOKEN` | 8007 | **[secret]** aura-copilot MCP server |

### Memory & Storage

| Variable | Default | Description |
|---|---|---|
| `AURA_DB_PATH` | `./memory` | Base directory for all persistent state |
| `AURA_DB_URL` | `sqlite:///./memory/aura.db` | Full SQLite connection URL |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `REDIS_ENABLED` | `false` | Explicit opt-in to Redis (requires reachable `REDIS_URL`) |

### Feature Flags

| Variable | Default | Description |
|---|---|---|
| `AURA_ENABLE_NEW_ORCHESTRATOR` | `true` | Enable the multi-agent orchestrator |
| `AURA_FORCE_LEGACY_ORCHESTRATOR` | `false` | Force legacy single-agent loop as a rollback escape hatch |
| `AURA_ENABLE_SHADOW_MODE` | `false` | [EXPERIMENTAL] Run both orchestrators; return legacy results |
| `AURA_ENABLE_MCP_REGISTRY` | `true` | Enable MCP service registry at `/discovery` |
| `AURA_STRICT_SCHEMA` | `false` | [EXPERIMENTAL] Strict Pydantic validation on tool I/O |
| `AURA_ENABLE_SWARM` | `0` | [EXPERIMENTAL] Autonomous parallel agent fleet (requires Redis) |

### Security

| Variable | Default | Description |
|---|---|---|
| `AURA_SECRET_KEY` | — | **[secret] [required in prod]** Internal session token signing key |
| `JWT_SECRET_KEY` | — | **[secret]** JWT signing key when `AUTH_ENABLED=true` |
| `AUTH_ENABLED` | `false` | Enable bearer-token auth on all API endpoints |
| `AURA_AUTO_PROVISION_MCP` | `false` | Auto-provision missing MCP servers on startup |

### External Services

| Variable | Description |
|---|---|
| `GITHUB_TOKEN` | **[secret]** GitHub PAT for repo operations and the GitHub MCP bridge |
| `SENTRY_DSN` | **[secret]** Sentry DSN for error tracking |
| `BRAVE_API_KEY` | **[secret]** Brave Search API key |
| `AURA_N8N_ENABLED` | Enable n8n integration (default `false`) |
| `N8N_WEBHOOK_URL` | Base URL of the n8n instance |
| `PROMETHEUS_ENABLED` | Enable `/metrics` scraping (default `false`) |

---

## 3 · aura.config.json — Key Fields

Full schema defined in `core/config_schema.py` (`AuraConfig`).

| Field | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `google/gemini-2.0-flash-exp:free` | Default model for uncategorised tasks |
| `max_iterations` | `int` (1–1000) | `10` | Max agent loop iterations per goal |
| `max_cycles` | `int` (1–100) | `5` | Max policy cycles per run |
| `dry_run` | `bool` | `false` | Simulate execution without side-effects |
| `decompose` | `bool` | `false` | Auto-decompose complex goals into sub-tasks |
| `strict_schema` | `bool` | `false` | Strict tool I/O validation |
| `policy_name` | `"sliding_window"` \| `"fixed"` \| `"unlimited"` | `sliding_window` | Execution policy |
| `policy_max_cycles` | `int` | `5` | Max cycles under the policy |
| `policy_max_seconds` | `int` (1–3600) | `120` | Max wall-clock seconds under the policy |
| `enable_new_orchestrator` | `bool` | `true` | Enable multi-agent orchestrator |
| `auto_add_capabilities` | `bool` | `true` | Auto-register new agent capabilities |
| `auto_backfill_coverage` | `bool` | `true` | Auto-generate missing test coverage |
| `llm_timeout` | `int` (1–600) | `60` | LLM call timeout in seconds |
| `mcp_server_url` | `str` | `http://localhost:8000` | Base URL of the MCP server |

### Nested — `model_routing`

```jsonc
"model_routing": {
  "code_generation": "google/gemini-2.0-flash-exp:free",
  "planning":        "google/gemini-2.0-flash-exp:free",
  "analysis":        "google/gemini-2.0-flash-exp:free",
  "critique":        "google/gemini-2.0-flash-exp:free",
  "embedding":       "openai/text-embedding-3-small",
  "fast":            "google/gemini-2.0-flash-exp:free",
  "quality":         "anthropic/claude-3.5-sonnet"
}
```

Each value can be a single model ID string or a list of model IDs (one is
chosen at random per invocation for load-balancing).

### Nested — `beads`

```jsonc
"beads": {
  "enabled": true,
  "required": true,
  "bridge_command": null,       // path to beads_bridge.mjs if non-default
  "timeout_seconds": 20,        // 1–300
  "scope": "goal_run",          // "goal_run" | "session" | "project"
  "persist_artifacts": true
}
```

### Nested — `semantic_memory`

```jsonc
"semantic_memory": {
  "enabled": true,
  "backend": "sqlite_local",    // "sqlite_local" | "qdrant"
  "embedding_model": "text-embedding-3-small",
  "top_k": 10,                  // 1–100
  "min_score": 0.65,            // 0.0–1.0
  "max_snippet_chars": 2000     // 100–10000
}
```

### Nested — `mcp_servers` (port registry)

```jsonc
"mcp_servers": {
  "dev_tools":    8001,
  "skills":       8002,
  "control":      8003,
  "agentic_loop": 8006,
  "copilot":      8007
}
```

---

## 4 · settings.json — Key Fields

`settings.json` shares the same schema as `aura.config.json`.  It is loaded
as a lower-priority fallback (below `aura.config.json`, above built-in
defaults).  Typical use: VS Code extension stores user preferences here.

---

## 5 · .env.example

A fully-annotated `.env.example` is provided at the repository root.  To
bootstrap a new environment:

```bash
cp .env.example .env
# Then fill in secrets:
#   OPENROUTER_API_KEY, AGENT_API_TOKEN, JWT_SECRET_KEY, AURA_SECRET_KEY
```

> **Never commit `.env` to version control.**  It contains secrets.  
> `.env.example` is safe to commit — it contains only placeholders.

---

## 6 · Validating Configuration

Use `scripts/validate_config.py` to check your current configuration against
the Pydantic schema before starting the server:

```bash
# Validate aura.config.json in the current directory:
python3 scripts/validate_config.py

# Validate an explicit config file:
python3 scripts/validate_config.py --config /path/to/aura.config.json

# Show the merged effective config (all layers resolved):
python3 scripts/validate_config.py --show-effective
```

Exit code `0` means valid; non-zero means validation errors were found and
printed to stderr.

### Inline validation (Python)

```python
from core.config_schema import validate_config

is_valid, errors = validate_config({"max_iterations": 5, "dry_run": True})
if not is_valid:
    for err in errors:
        print(err)
```

---

## 7 · OpenAPI Spec Export

The FastAPI server exposes its schema at `/openapi.json` when running.
To export it statically (e.g. for offline validation or SDK generation):

```bash
AURA_TEST_MODE=1 python3 scripts/export_openapi.py
# → docs/api/openapi.json
```

The spec is validated on each CI run via `openapi_spec_validator`.
