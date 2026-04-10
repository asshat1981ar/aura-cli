# AURA CLI

[![CI](https://github.com/asshat1981ar/aura-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/asshat1981ar/aura-cli/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![Version](https://img.shields.io/badge/version-1.0.0-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Autonomous software development platform. AURA accepts natural-language goals and runs a **10-phase multi-agent pipeline** (ingest → plan → critique → code → apply → verify → reflect → adapt → evolve → archive) to design, implement, test, and commit changes with minimal human intervention.

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/asshat1981ar/aura-cli.git
cd aura-cli
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set at minimum:
#   AURA_JWT_SECRET=<random 43-char token>
#   OPENAI_API_KEY=<key>   (or ANTHROPIC_API_KEY)
```

Generate a secret:
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(43))"
```

### 3. Run once (single goal)

```bash
aura goal once "Add input validation to the login endpoint"
# or
./run_aura.sh once "Add input validation to the login endpoint"
# or
python3 main.py goal once "Add input validation to the login endpoint"
```

### 4. Run loop (continuous goal queue)

```bash
aura goal run
# or
./run_aura.sh run
```

### 5. Start the API server

```bash
uvicorn aura_cli.server:app --host 0.0.0.0 --port 8001
# Interactive docs at http://localhost:8001/docs
```

### 6. Docker

```bash
# Development stack
docker compose up

# Production stack
docker compose -f docker-compose.prod.yml up
```

---

## Entry Points Decision Tree

| I need to… | Use |
|---|---|
| Run one goal and exit | `aura goal once "<goal>"` |
| Drain the persistent goal queue | `aura goal run` |
| Add a goal to the queue | `aura goal add "<goal>"` |
| Check queue / last run status | `aura goal status` |
| List running agents | `aura agent list` |
| Start the REST API + WebSocket server | `uvicorn aura_cli.server:app --port 8001` |
| Use the interactive CLI shell | `aura` (no subcommand) |
| Use the shell wrapper (alias-friendly) | `./run_aura.sh <command>` |
| Use the developer shim | `python3 main.py <command>` |
| List / call MCP tools | `aura mcp tools` / `aura mcp call <tool> '<json>'` |
| Search memory | `aura memory search "<query>"` |
| Inspect health | `curl http://localhost:8001/health` |

---

## Architecture

```
Natural-language goal
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│                   10-Phase Multi-Agent Pipeline                   │
│                                                                   │
│  ingest → plan → critique → code → apply →                       │
│  verify → reflect → adapt → evolve → archive                     │
└───────────────┬───────────────────────────────┬──────────────────┘
                │                               │
                ▼                               ▼
┌───────────────────────────┐    ┌──────────────────────────────┐
│       Memory Tiers        │    │         MCP Servers          │
│                           │    │                              │
│  • SQLite (goal history)  │    │  :8001  aura_cli.server      │
│  • JSONL (decision log)   │    │  :8002  aura_mcp_skills      │
│  • Redis (optional cache) │    │  :8007  github_copilot_mcp   │
│  • Semantic index         │    │  :8020  sadd_mcp_server      │
└───────────────────────────┘    │  :8030  observability_mcp    │
                                 └──────────────────────────────┘
```

---

## Core Modules

| Path | Role |
|---|---|
| `aura_cli/cli_main.py` | Console entry point (`aura` script) |
| `aura_cli/server.py` | FastAPI application — all REST + WebSocket routes |
| `aura_cli/middleware/` | Rate-limiting, auth, CORS middleware |
| `core/` | ~120 orchestration modules (auth, config, sanitizer, correlation…) |
| `core/file_tools.py` | Autonomous apply safety — overwrite policy enforcement |
| `core/sanitizer.py` | Input validation and sanitization |
| `agents/` | Specialized pipeline agents (planner, coder, sandbox, applicator…) |
| `agents/sandbox.py` | Sandboxed subprocess execution with tempdir isolation |
| `memory/` | Persistence layer (SQLite, JSONL decision log, Redis cache) |
| `tools/mcp_server.py` | Primary MCP server |
| `tools/sadd_mcp_server.py` | Sub-Agent Driven Development MCP server |
| `tools/observability_mcp.py` | Observability MCP server |
| `main.py` | Lightweight developer shim → delegates to `aura_cli.cli_main:main` |
| `run_aura.sh` | Shell wrapper with convenience aliases |

---

## Configuration

### Precedence (highest → lowest)

| Layer | Source |
|---|---|
| 1 (highest) | Environment variables / `.env` |
| 2 | `aura.config.json` (or path in `AURA_CONFIG_PATH`) |
| 3 | `settings.json` (model routing, provider config) |
| 4 (lowest) | Built-in defaults |

### Key environment variables

| Variable | Default | Description |
|---|---|---|
| `AURA_ENV` | `development` | Runtime environment (`development`\|`staging`\|`production`) |
| `AURA_LOG_LEVEL` | `info` | Log verbosity (`debug`\|`info`\|`warn`\|`error`) |
| `AURA_JWT_SECRET` | — | **[required, secret]** JWT signing key |
| `AURA_JWT_EXPIRY` | `24h` | Token lifetime |
| `AURA_API_HOST` | `0.0.0.0` | Server bind address |
| `AURA_API_PORT` | `8001` | Server bind port |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `REDIS_ENABLED` | `false` | Opt-in to Redis caching |
| `AURA_DRY_RUN` | `false` | Simulate execution (no LLM calls, no writes) |
| `AURA_SKIP_CHDIR` | unset | Set to `1` when running tests from repo root |
| `AURA_TEST_MODE` | unset | Set to `1` to enable JWT `algorithm=none` test bypass |
| `AURA_CONFIG_PATH` | auto | Explicit path to config file |

### Key config files

| File | Purpose |
|---|---|
| `.env` | Runtime secrets and overrides (never commit) |
| `aura.config.json` | Project config (model, API keys, cycle settings) |
| `settings.json` | Model routing and provider selection |
| `.mcp.json` | Repo-local MCP server registry |
| `aura_auth.db` | JWT revocation store — SQLite (never commit) |

---

## Security

AURA enforces multiple security layers:

- **JWT authentication** — all API endpoints require a `Bearer` token; tokens are revocable via `aura_auth.db`.
- **Rate limiting** — token-bucket middleware on every HTTP request; headers expose current limits.
- **Sandbox isolation** — untrusted code runs in a subprocess with a private temp directory; violations are counted in Prometheus (`aura_sandbox_violations_total`).
- **Input sanitization** — all user-controlled strings pass through `core/sanitizer.py` before use.
- **Autonomous apply policy** — stale-snippet overwrites are blocked by default (`core/file_tools.py`).

Auth DB path: `aura_auth.db` (SQLite, in repo root — **never commit**).

See [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) for the full threat model, attack surface analysis, and mitigations.

---

## API

The server runs at `http://localhost:8001` by default. Interactive Swagger docs are available at **`/docs`**.

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | — | Liveness probe |
| `GET` | `/ready` | — | Readiness probe (checks subsystems) |
| `GET` | `/metrics` | — | Prometheus metrics (scraper network only) |
| `GET` | `/tools` | JWT | List available MCP tools |
| `GET` | `/discovery` | JWT | MCP discovery payload |
| `GET` | `/environments` | JWT | List registered AI environments |
| `GET` | `/architecture` | JWT | Default MCP routing profile |
| `POST` | `/execute` | JWT | Execute a named tool |
| `POST` | `/run` | JWT | Trigger a full pipeline run via `LoopOrchestrator` |
| `POST` | `/webhook/goal` | JWT | Enqueue a goal (n8n / CI trigger) |
| `GET` | `/webhook/status/{goal_id}` | JWT | Poll webhook-submitted goal status |
| `POST` | `/webhook/plan-review` | JWT | Format task bundle for quality-gate review |
| `GET` | `/api/health` | — | Router health (alias) |

WebSocket and run-management routes are registered under `/api/`.

---

## Testing

> ⚠️ **WARNING**: Never run `pytest tests/` without a timeout — some test files hang indefinitely. Always use the safe suite below with `--timeout=30`.

### Fast regression suite (~4–6 s)

```bash
python3 -m pytest \
  tests/test_auth.py \
  tests/test_jwt_hardening.py \
  tests/test_server_api.py \
  tests/test_cli_exit_codes.py \
  tests/test_sanitizer.py \
  tests/test_correlation.py \
  tests/test_config_schema.py \
  tests/test_redis_cache.py \
  tests/test_sandbox_violations.py \
  tests/test_sandbox_unit.py \
  tests/test_e2e_sandbox_retry.py \
  -v --timeout=30 --no-cov
```

### With coverage

```bash
python3 -m pytest tests/test_auth.py tests/test_sanitizer.py \
  -v --timeout=30 \
  --cov=aura_cli --cov=core --cov=agents --cov=memory \
  --cov-report=term-missing
```

### Triage all test files (finds hangers)

```bash
python3 scripts/triage_tests.py --timeout 30
# Reports written to: reports/test-triage.json, reports/safe-tests.txt
```

Required env vars for tests:
```bash
export AURA_SKIP_CHDIR=1
export AURA_TEST_MODE=1
```

---

## Development

### Pre-commit hooks

```bash
pre-commit install
pre-commit run --all-files
```

### Lint

```bash
python3 -m ruff check .
```

### Regenerate CLI reference docs

```bash
python3 scripts/generate_cli_reference.py
# Verify docs are current (CI check):
python3 scripts/generate_cli_reference.py --check
```

### Other utilities

```bash
# Validate config files
python3 scripts/validate_config.py

# Find highest-impact modules to add coverage for
python3 scripts/find_coverage_gaps.py --top 10
```

---

## Docker

```bash
# Development stack (includes n8n, observability)
docker compose up

# Production stack
docker compose -f docker-compose.prod.yml up

# n8n workflow stack only
docker compose -f docker-compose.n8n.yml up

# Observability stack only
docker compose -f docker-compose.observability.yml up
```

---

## Documentation

| Document | Description |
|---|---|
| [`docs/CLI_REFERENCE.md`](docs/CLI_REFERENCE.md) | Generated CLI command reference |
| [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) | Security threat model and mitigations |
| [`docs/API_GUIDE.md`](docs/API_GUIDE.md) | REST API guide |
| [`docs/API.md`](docs/API.md) | API reference |
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | End-user guide |
| [`docs/CONTRIBUTING.md`](CONTRIBUTING.md) | Contributor guide |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Deployment guide |
| [`docs/MEMORY_ARCHITECTURE.md`](docs/MEMORY_ARCHITECTURE.md) | Memory tier architecture |
| [`docs/INTEGRATION_MAP.md`](docs/INTEGRATION_MAP.md) | Integration map |
| [`docs/MCP_SERVERS.md`](docs/MCP_SERVERS.md) | MCP server reference |
| [`docs/INNOVATION_CATALYST.md`](docs/INNOVATION_CATALYST.md) | Structured brainstorming guide |
| [`docs/AURA_ITERATIVE_WORKFLOW.md`](docs/AURA_ITERATIVE_WORKFLOW.md) | Iterative workflow guide |
| [`docs/AURA_MULTI_AGENT_WORKFLOW.md`](docs/AURA_MULTI_AGENT_WORKFLOW.md) | Multi-agent workflow guide |
| [`docs/AURA_OPERATOR_PROMPT.md`](docs/AURA_OPERATOR_PROMPT.md) | Operator prompt reference |
| [`docs/DEVELOPMENT_AUTOMATION_GUIDE.md`](docs/DEVELOPMENT_AUTOMATION_GUIDE.md) | Development automation guide |
| [`docs/SECURITY.md`](docs/SECURITY.md) | Security policy |
| [`docs/SECRET_MANAGEMENT.md`](docs/SECRET_MANAGEMENT.md) | Secret management guide |
| [`docs/ROADMAP_PRD_SERIES.md`](docs/ROADMAP_PRD_SERIES.md) | Roadmap and PRD series |
| [`docs/TECH_DEBT.md`](docs/TECH_DEBT.md) | Known technical debt |
| [`docs/WEB_UI_GUIDE.md`](docs/WEB_UI_GUIDE.md) | Web UI dashboard guide |

---

## License

MIT — see [LICENSE](LICENSE).

