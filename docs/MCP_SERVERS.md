# AURA MCP Servers Reference

AURA exposes **6 Model Context Protocol (MCP) HTTP servers** plus the
`playwright` stdio MCP helper. An optional `n8n-mcp` integration can be merged
into local configs alongside the AURA-managed servers.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Authentication](#authentication)
3. [Health Check](#health-check)
4. [VS Code / Copilot CLI Integration](#vs-code--copilot-cli-integration)
5. [Server Reference](#server-reference)
   - [aura-dev-tools (port 8001)](#1-aura-dev-tools--port-8001)
   - [aura-skills (port 8002)](#2-aura-skills--port-8002)
   - [aura-control (port 8003)](#3-aura-control--port-8003)
   - [aura-agentic-loop (port 8006)](#4-aura-agentic-loop--port-8006)
   - [aura-copilot (port 8007)](#5-aura-copilot--port-8007)
   - [aura-sadd (port 8020)](#6-aura-sadd--port-8020)
   - [n8n-mcp (port 5678)](#7-n8n-mcp--port-5678)
6. [aura-skills: All 35 Skills](#aura-skills-all-35-skills)

---

## Quick Start

```bash
# 1. Load auth tokens
source .env.n8n          # sets AGENT_API_TOKEN, MCP_API_TOKEN, SADD_MCP_TOKEN, etc.

# 2. Start servers (each in its own terminal, or use the docker-compose stack)
uvicorn aura_cli.server:app --host 127.0.0.1 --port 8001 &
uvicorn tools.aura_mcp_skills_server:app --host 127.0.0.1 --port 8002 &
uvicorn tools.aura_control_mcp:app --host 127.0.0.1 --port 8003 &
uvicorn tools.agentic_loop_mcp:app --host 127.0.0.1 --port 8006 &
uvicorn tools.github_copilot_mcp:app --host 127.0.0.1 --port 8007 &
uvicorn tools.sadd_mcp_server:app --host 127.0.0.1 --port 8020 &

# 3. Verify all are running
for port in 8001 8002 8003 8006 8007 8020; do
  curl -sf http://localhost:$port/health \
    && echo "port $port OK" \
    || echo "port $port DOWN"
done
```

Alternatively, use the Docker Compose stack which starts all services together:

```bash
docker compose up -d
```

---

## Authentication

All six AURA MCP servers use **Bearer token** authentication. Each server reads
its token from a dedicated environment variable. If the variable is empty, auth
is skipped (development / local mode).

| Server           | Port | Environment Variable  | Header                            |
|------------------|------|-----------------------|-----------------------------------|
| aura-dev-tools   | 8001 | `AGENT_API_TOKEN`     | `Authorization: Bearer <token>`   |
| aura-skills      | 8002 | `MCP_API_TOKEN`       | `Authorization: Bearer <token>`   |
| aura-control     | 8003 | `MCP_CONTROL_TOKEN`   | `Authorization: Bearer <token>` **or** `X-API-Key: <token>` |
| aura-agentic-loop| 8006 | `AGENTIC_LOOP_TOKEN`  | `Authorization: Bearer <token>` **or** `X-API-Key: <token>` |
| aura-copilot     | 8007 | `COPILOT_MCP_TOKEN`   | `Authorization: Bearer <token>` **or** `X-API-Key: <token>` |
| aura-sadd        | 8020 | `SADD_MCP_TOKEN`      | `Authorization: Bearer <token>`   |
| n8n-mcp          | 5678 | `N8N_MCP_TOKEN`       | `Authorization: Bearer <token>`   |

Define tokens in `.env.n8n` (never commit this file):

```dotenv
AGENT_API_TOKEN=your-dev-tools-token
MCP_API_TOKEN=your-skills-token
MCP_CONTROL_TOKEN=your-control-token
AGENTIC_LOOP_TOKEN=your-loop-token
COPILOT_MCP_TOKEN=your-copilot-token
SADD_MCP_TOKEN=your-sadd-token
N8N_MCP_TOKEN=your-n8n-token
```

---

## Health Check

All servers expose `GET /health` (no auth required):

```bash
# Single server
curl http://localhost:8001/health

# All servers at once
for port in 8001 8002 8003 8006 8007; do
  status=$(curl -sf http://localhost:$port/health 2>&1 || echo '{"status":"DOWN"}')
  echo "port $port: $status"
done
```

Expected response shape:

```json
{
  "status": "ok",
  "server": "aura-dev-tools",
  "version": "1.0.0",
  "uptime_s": 123.4
}
```

---

## VS Code / Copilot CLI Integration

MCP servers are wired into VS Code and Copilot CLI via `.vscode/mcp.json`. A
safe example is committed as `.vscode/mcp.json.example`; copy it and fill in
your tokens:

```bash
cp .vscode/mcp.json.example .vscode/mcp.json
# Edit .vscode/mcp.json and replace ${env:...} references if needed,
# OR ensure the env vars are exported in your shell before launching VS Code.
```

The generated config registers all six AURA HTTP endpoints plus `playwright`
and references tokens via `${env:VAR_NAME}` so secrets are never hardcoded:

```jsonc
{
  "mcpServers": {
    "aura-dev-tools":    { "type": "http", "url": "http://127.0.0.1:8001",
                           "headers": { "Authorization": "Bearer ${env:AGENT_API_TOKEN}" } },
    "aura-skills":       { "type": "http", "url": "http://127.0.0.1:8002",
                           "headers": { "Authorization": "Bearer ${env:MCP_API_TOKEN}" } },
    "aura-control":      { "type": "http", "url": "http://127.0.0.1:8003",
                           "headers": { "Authorization": "Bearer ${env:MCP_CONTROL_TOKEN}" } },
    "aura-agentic-loop": { "type": "http", "url": "http://127.0.0.1:8006",
                           "headers": { "Authorization": "Bearer ${env:AGENTIC_LOOP_TOKEN}" } },
    "aura-copilot":      { "type": "http", "url": "http://127.0.0.1:8007",
                            "headers": { "Authorization": "Bearer ${env:COPILOT_MCP_TOKEN}" } },
    "aura-sadd":         { "type": "http", "url": "http://127.0.0.1:8020",
                            "headers": { "Authorization": "Bearer ${env:SADD_MCP_TOKEN}" } },
    "playwright":        { "type": "stdio", "command": "npx",
                            "args": ["@playwright/mcp@latest"] }
  }
}
```

> **Tip:** Reload the MCP config in Copilot CLI after editing: run
> `/mcp reload` or restart VS Code.

If you also use n8n locally, merge this additional unmanaged entry into your
config:

```jsonc
"n8n-mcp": {
  "type": "http",
  "url": "http://localhost:5678/mcp-server/http",
  "headers": {
    "Authorization": "Bearer ${env:N8N_MCP_TOKEN}",
    "Accept": "application/json, text/event-stream"
  }
}
```

---

## Server Reference

### 1. aura-dev-tools — Port 8001

**Source:** `aura_cli/server.py`  
**Token:** `AGENT_API_TOKEN`  
**Title:** AURA Dev Tools MCP  
**Purpose:** Main AURA orchestration entry-point. Exposes shell execution,
goal-running, and n8n pipeline webhooks. Lazily initialises the full AURA
runtime (orchestrator, model adapter, memory store) on first request.

#### Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET`  | `/health` | ❌ | Health check + provider status |
| `GET`  | `/metrics` | ✅ | Runtime metrics: total calls, registered services, run-tool audit |
| `GET`  | `/tools` | ✅ | List available MCP tools |
| `GET`  | `/discovery` | ✅ | Discover all registered AURA services and supported AI environments |
| `GET`  | `/environments` | ✅ | List detected AI environments (conda, venv, pyenv, etc.) |
| `GET`  | `/architecture` | ✅ | Routing profile + knowledge backends + environment list |
| `POST` | `/execute` | ✅ | Invoke a tool by name |
| `POST` | `/webhook/goal` | ✅ | Enqueue a goal from n8n (returns `goal_id`) |
| `GET`  | `/webhook/status/{goal_id}` | ✅ | Poll status of a webhook-submitted goal |
| `POST` | `/webhook/plan-review` | ✅ | Format a task bundle for Dev Suite quality-gate review |

#### Tools (`POST /execute`)

| `tool_name` | Description |
|-------------|-------------|
| `ask`  | Ask AURA a question; streams response via the configured model adapter |
| `run`  | Run a shell command with streaming SSE output (requires `AGENT_API_ENABLE_RUN=1`) |
| `goal` | Execute an autonomous goal cycle via the orchestrator; SSE-streamed |
| `env`  | *(Disabled — returns 501 for security reasons)* |

**Environment controls for `run` tool:**

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_API_ENABLE_RUN` | `""` (disabled) | Set to `1` to enable shell execution |
| `AURA_RUN_TOOL_TIMEOUT_S` | `15` | Per-command timeout in seconds (max 60) |
| `AURA_RUN_TOOL_MAX_OUTPUT_BYTES` | `65536` | Output truncation limit (max 262144) |
| `AURA_RUN_TOOL_READ_CHUNK_BYTES` | `1024` | Streaming chunk size (max 8192) |

**Example — invoke `ask`:**

```bash
curl -X POST http://localhost:8001/execute \
  -H "Authorization: Bearer $AGENT_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "ask", "args": ["What is the current project status?"]}'
```

**Example — enqueue a goal via webhook:**

```bash
curl -X POST http://localhost:8001/webhook/goal \
  -H "Authorization: Bearer $AGENT_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"goal": "Refactor auth module for cleaner error handling", "priority": 5}'
# → {"status": "queued", "goal_id": "abc123..."}

# Poll for completion:
curl http://localhost:8001/webhook/status/abc123... \
  -H "Authorization: Bearer $AGENT_API_TOKEN"
```

---

### 2. aura-skills — Port 8002

**Source:** `tools/aura_mcp_skills_server.py`  
**Token:** `MCP_API_TOKEN`  
**Title:** AURA MCP Skills Server  
**Purpose:** Exposes all 35 registered AURA skill modules as individual MCP
tools. Skills perform static analysis, code generation, security scanning,
and more — all without requiring a live LLM call.

#### Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET`  | `/health` | ❌ | Health + loaded skill count |
| `GET`  | `/tools` | ✅ | List all 35 skills as MCP tool descriptors |
| `POST` | `/call` | ✅ | Invoke a skill by name with an args dict |
| `GET`  | `/skill/{name}` | ✅ | Descriptor for a single skill |

**Example — list skills:**

```bash
curl http://localhost:8002/tools \
  -H "Authorization: Bearer $MCP_API_TOKEN" | python3 -m json.tool
```

**Example — run the security scanner:**

```bash
curl -X POST http://localhost:8002/call \
  -H "Authorization: Bearer $MCP_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "security_scanner",
    "args": {"project_root": "."}
  }'
```

See [All 35 Skills](#aura-skills-all-35-skills) below for the complete list.

---

### 3. aura-control — Port 8003

**Source:** `tools/aura_control_mcp.py`  
**Token:** `MCP_CONTROL_TOKEN`  
**Title:** AURA Control MCP Server  
**Purpose:** Control plane for the AURA autonomous development loop. Manage
goals, search memory/brain, and read project files — all safely jailed to the
project root.

#### Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET`  | `/health` | ❌ | Health check + uptime |
| `GET`  | `/tools` | ✅ | List all control tools as MCP descriptors |
| `POST` | `/call` | ✅ | Invoke a control tool by name |
| `GET`  | `/tool/{name}` | ✅ | Descriptor for a single tool |

#### Tools (`POST /call`)

**Goal management:**

| Tool | Required args | Description |
|------|---------------|-------------|
| `goal_add` | `text: string` | Enqueue a new goal |
| `goal_list` | — | List all queued goals |
| `goal_remove` | `index: int` | Remove goal at zero-based index |
| `goal_clear` | — | Remove all goals (irreversible) |
| `goal_archive_list` | `limit?: int` | List completed/archived goals with outcomes |

**Memory / Brain:**

| Tool | Required args | Description |
|------|---------------|-------------|
| `memory_search` | `query: string`, `limit?: int` | Keyword search of the brain store (30s cache) |
| `memory_add` | `text: string` | Store a new memory |
| `memory_weaknesses` | — | List recorded agent weaknesses |

**File access (project-root jailed):**

| Tool | Required args | Description |
|------|---------------|-------------|
| `file_read` | `path: string` (relative) | Read a file within the project root |
| `file_list` | `path?: string` (relative) | List files/directories within the project root |

**Meta:**

| Tool | Required args | Description |
|------|---------------|-------------|
| `project_status` | — | Summary: queue size, memory count, archive size |

**Example — add a goal:**

```bash
curl -X POST http://localhost:8003/call \
  -H "Authorization: Bearer $MCP_CONTROL_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "goal_add", "args": {"text": "Write unit tests for core/config_manager.py"}}'
```

**Example — search memory:**

```bash
curl -X POST http://localhost:8003/call \
  -H "Authorization: Bearer $MCP_CONTROL_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "memory_search", "args": {"query": "authentication", "limit": 10}}'
```

---

### 4. aura-agentic-loop — Port 8006

**Source:** `tools/agentic_loop_mcp.py`  
**Token:** `AGENTIC_LOOP_TOKEN`  
**Title:** Agentic Loop MCP Server  
**Purpose:** Exposes AURA's workflow engine and agentic loop as MCP tools.
Supports defining multi-step skill workflows and running autonomous goal
processing loops with full lifecycle control (pause/resume/cancel/health).

#### Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET`  | `/health` | ❌ | Health + engine stats |
| `GET`  | `/metrics` | ✅ | Call counts + error rates per tool |
| `GET`  | `/tools` | ✅ | List all tools as MCP descriptors |
| `POST` | `/call` | ✅ | Invoke a tool by name |
| `GET`  | `/tool/{name}` | ✅ | Descriptor for a single tool |
| `GET`  | `/workflows` | ✅ | Shortcut: list workflow definitions |
| `GET`  | `/loops` | ✅ | Shortcut: list all loops |

#### Tools (`POST /call`)

**Workflow management:**

| Tool | Required args | Description |
|------|---------------|-------------|
| `workflow_define` | `name`, `steps[]` | Register a custom ordered skill workflow |
| `workflow_list` | — | List all registered workflow definitions |
| `workflow_run` | `workflow_name`, `inputs?`, `background?` | Start a workflow; returns `exec_id` |
| `workflow_status` | `exec_id` | Live status + step history |
| `workflow_output` | `exec_id`, `step_name` | Full output dict for a completed step |
| `workflow_cancel` | `exec_id` | Cancel a running or paused execution |
| `workflow_pause` | `exec_id` | Pause after current step |
| `workflow_resume` | `exec_id` | Resume from where it stopped |

**Agentic loop management:**

| Tool | Required args | Description |
|------|---------------|-------------|
| `loop_create` | `goal`, `max_cycles?` | Create a new goal-processing loop |
| `loop_tick` | `loop_id`, `dry_run?` | Execute one loop cycle |
| `loop_status` | `loop_id` | Full state: cycle history, stop reason, elapsed time |
| `loop_stop` | `loop_id`, `reason?` | Stop immediately |
| `loop_pause` | `loop_id` | Pause after current cycle |
| `loop_resume` | `loop_id` | Resume a paused loop |
| `loop_health` | `loop_id`, `stall_threshold_s?` | Detect stalls, deadlocks, repeated errors |
| `loop_list` | `status?` | List all loops (optionally filtered by status) |

**Example — run a workflow:**

```bash
# Define a code-quality workflow
curl -X POST http://localhost:8006/call \
  -H "Authorization: Bearer $AGENTIC_LOOP_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "workflow_define",
    "args": {
      "name": "code-quality",
      "steps": [
        {"name": "lint",     "skill_name": "linter_enforcer",   "static_inputs": {"project_root": "."}},
        {"name": "security", "skill_name": "security_scanner",  "static_inputs": {"project_root": "."}},
        {"name": "debt",     "skill_name": "tech_debt_quantifier", "static_inputs": {"project_root": "."}}
      ]
    }
  }'

# Run it
curl -X POST http://localhost:8006/call \
  -H "Authorization: Bearer $AGENTIC_LOOP_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "workflow_run", "args": {"workflow_name": "code-quality", "background": true}}'
# → {"exec_id": "wf-abc123"}

# Poll status
curl -X POST http://localhost:8006/call \
  -H "Authorization: Bearer $AGENTIC_LOOP_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "workflow_status", "args": {"exec_id": "wf-abc123"}}'
```

**Example — create and run an agentic loop:**

```bash
# Create loop
curl -X POST http://localhost:8006/call \
  -H "Authorization: Bearer $AGENTIC_LOOP_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "loop_create", "args": {"goal": "Fix all type errors in aura_cli/", "max_cycles": 10}}'
# → {"loop_id": "loop-xyz"}

# Tick it forward (call until loop_status shows terminal)
curl -X POST http://localhost:8006/call \
  -H "Authorization: Bearer $AGENTIC_LOOP_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "loop_tick", "args": {"loop_id": "loop-xyz"}}'
```

---

### 5. aura-copilot — Port 8007

**Source:** `tools/github_copilot_mcp.py`  
**Token:** `COPILOT_MCP_TOKEN`  
**Title:** GitHub Copilot MCP  
**Purpose:** Copilot Workspace-style AI intelligence over live GitHub data.
Combines the GitHub REST API with AURA's ModelAdapter to provide AI-powered
issue analysis, PR review, code explanation, test generation, and repo health.

**Required environment variables:**

| Variable | Description |
|----------|-------------|
| `GITHUB_PAT` | GitHub Personal Access Token (repo + read:org scopes) |
| `AURA_API_KEY` | OpenRouter / OpenAI key used by ModelAdapter |

#### Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET`  | `/health` | ❌ | Health + model/GitHub connectivity |
| `GET`  | `/tools` | ✅ | List all Copilot tools |
| `POST` | `/call` | ✅ | Invoke a tool by name |
| `GET`  | `/tool/{name}` | ✅ | Descriptor for a single tool |

#### Tools (`POST /call`)

| Tool | Required args | Description |
|------|---------------|-------------|
| `issue_analyze` | `repo`, `issue_number` | Root-cause analysis + numbered implementation plan |
| `pr_review` | `repo`, `pr_number` | AI code review with severity-tagged findings |
| `pr_describe` | `repo`, `pr_number` | Auto-generate PR title + description from diff |
| `code_explain` | `code` | Explain a snippet in plain English |
| `code_fix` | `code`, `error` | Suggest a targeted fix for a bug |
| `test_generate` | `code` | Generate happy-path, edge-case, and error tests |
| `commit_message` | `diff` | Generate a Conventional Commit message |
| `repo_health` | `repo` | Issue age, stale PRs, activity summary |
| `issue_to_plan` | `repo`, `issue_number` | Detailed implementation plan from an issue |
| `find_related_code` | `repo`, `query` | Search + AI-rank repo code by relevance |

**Example — analyze an issue:**

```bash
curl -X POST http://localhost:8007/call \
  -H "Authorization: Bearer $COPILOT_MCP_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "issue_analyze",
    "args": {"repo": "myorg/myrepo", "issue_number": 42}
  }'
```

**Example — generate a commit message:**

```bash
DIFF=$(git diff HEAD~1)
curl -X POST http://localhost:8007/call \
  -H "Authorization: Bearer $COPILOT_MCP_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"tool_name\": \"commit_message\", \"args\": {\"diff\": $(echo "$DIFF" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'), \"scope\": \"auth\"}}"
```

**Example — get repo health:**

```bash
curl -X POST http://localhost:8007/call \
  -H "Authorization: Bearer $COPILOT_MCP_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "repo_health", "args": {"repo": "myorg/myrepo"}}'
```

---

### 6. aura-sadd — Port 8020

**Source:** `tools/sadd_mcp_server.py`  
**Token:** `SADD_MCP_TOKEN`  
**Title:** SADD MCP  
**Purpose:** Exposes SADD design-spec parsing and session-tracking operations as
MCP tools.

#### Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET`  | `/health` | ✅ | Health check |
| `GET`  | `/tools` | ✅ | List all SADD tools |
| `POST` | `/call` | ✅ | Invoke a SADD tool by name |
| `GET`  | `/tool/{name}` | ✅ | Descriptor for a single tool |
| `GET`  | `/metrics` | ✅ | Per-tool call/error counts |

#### Tools (`POST /call`)

| Tool | Required args | Description |
|------|---------------|-------------|
| `sadd_parse_spec` | `spec_markdown` | Parse markdown spec text into workstreams |
| `sadd_session_status` | `session_id` | Get status for one SADD session |
| `sadd_list_sessions` | *(none)* | List recent SADD sessions |
| `sadd_session_events` | `session_id` | Fetch a session event log |
| `sadd_session_artifacts` | `session_id` | Fetch workstream artifacts |

**Example — parse a spec:**

```bash
curl -X POST http://localhost:8020/call \
  -H "Authorization: Bearer $SADD_MCP_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "sadd_parse_spec", "args": {"spec_markdown": "# Spec\n\nBuild a queue worker."}}'
```

---

### 7. n8n-mcp — Port 5678

**Source:** n8n workflow automation platform (see `docker-compose.n8n.yml`)  
**Token:** `N8N_MCP_TOKEN`  
**URL:** `http://localhost:5678/mcp-server/http`  

n8n is not an AURA-authored server but is registered in `.vscode/mcp.json`
because AURA's P1/P2/P3 pipelines run as n8n workflows. They call back into
`aura-dev-tools` via `POST /webhook/goal` and `POST /webhook/plan-review`.

Start n8n with:

```bash
docker compose -f docker-compose.n8n.yml up -d
```

---

## aura-skills: All 35 Skills

Exposed by **aura-skills (port 8002)**. Invoke any skill via `POST /call`
with `{"tool_name": "<name>", "args": {...}}`.

### Code Quality & Analysis

| Skill | Key args | What it does |
|-------|----------|--------------|
| `linter_enforcer` | `project_root`, `file_path`, `code` | flake8 style violations + AST naming checks |
| `complexity_scorer` | `project_root`, `code` | Cyclomatic complexity + nesting depth per function |
| `refactoring_advisor` | `project_root`, `code` | Detects god functions, deep nesting, magic numbers |
| `tech_debt_quantifier` | `project_root` | TODO/FIXME/HACK count + long file/function score (0–100) |
| `code_clone_detector` | `project_root`, `min_lines?`, `similarity_threshold?` | Exact (AST) + near-duplicate (Jaccard) clone detection |
| `incremental_differ` | `old_code`, `new_code` | Unified diff + added/removed symbols + impact analysis |

### Security & Compliance

| Skill | Key args | What it does |
|-------|----------|--------------|
| `security_scanner` | `project_root`, `code` | Hardcoded secrets, SQL injection, unsafe eval/exec/pickle |
| `security_hardener` | `project_root`, `code` | Suggests hardening fixes for detected vulnerabilities |

### Testing & Coverage

| Skill | Key args | What it does |
|-------|----------|--------------|
| `test_coverage_analyzer` | `project_root`, `min_target?` | Runs coverage.py and reports % with gap details |
| `generation_quality_checker` | `task`, `generated_code` | Scores AI-generated code quality without an LLM |
| `eval_optimizer` | (see descriptor) | Evaluator–optimizer loop for iterative quality improvement |

### Architecture & Structure

| Skill | Key args | What it does |
|-------|----------|--------------|
| `architecture_validator` | `project_root` | Import graph + circular import detection + coupling score |
| `dependency_analyzer` | `project_root` | Parses requirements; detects version conflicts and CVEs |
| `structural_analyzer` | `project_root` | High-level module structure and layering analysis |
| `api_contract_validator` | `code`, `old_spec?` | Extracts FastAPI/Flask routes; detects breaking changes |
| `schema_validator` | `schema`, `instance`, `code` | JSON Schema validation + Pydantic model discovery |
| `ast_analyzer` | (see descriptor) | Deep AST inspection for patterns and anti-patterns |

### Documentation & Generation

| Skill | Key args | What it does |
|-------|----------|--------------|
| `doc_generator` | `project_root`, `code` | AST-based docstring templates + README sections |
| `changelog_generator` | `project_root` | Generates CHANGELOG entries from git history |

### DevOps & Infra

| Skill | Key args | What it does |
|-------|----------|--------------|
| `dockerfile_analyzer` | `project_root` | Dockerfile best-practice checks and security findings |
| `observability_checker` | `project_root` | Checks for logging, metrics, tracing instrumentation |

### Performance & Profiling

| Skill | Key args | What it does |
|-------|----------|--------------|
| `performance_profiler` | `code` | cProfile + AST anti-pattern detection (nested loops etc.) |
| `database_query_analyzer` | (see descriptor) | Detects N+1 queries, missing indexes, slow patterns |

### Agentic & Workflow

| Skill | Key args | What it does |
|-------|----------|--------------|
| `skill_composer` | `goal`, `available_skills?` | Maps a natural-language goal to an ordered skill workflow |
| `adaptive_strategy_selector` | `goal`, `available_strategies?` | Recommends execution strategy from historical success rates |
| `error_pattern_matcher` | `current_error`, `error_history?` | Matches error to 12 known patterns; returns fix steps |
| `skill_failure_analyzer` | (see descriptor) | Root-cause analysis for skill failures |
| `evolution_skill` | (see descriptor) | Evolves skill implementations over time |
| `skill_generator` | (see descriptor) | Generates new skill scaffolding from a description |

### Tools & Utilities

| Skill | Key args | What it does |
|-------|----------|--------------|
| `type_checker` | `project_root`, `file_path` | Runs mypy or annotation-coverage heuristic |
| `git_history_analyzer` | `project_root`, `lookback_days?` | Hotspot files, change frequency, risky areas |
| `symbol_indexer` | (see descriptor) | Builds a symbol index for fast cross-file lookup |
| `multi_file_editor` | (see descriptor) | Applies coordinated multi-file edits |
| `web_fetcher` | `url` or `query` | Fetches a URL or DuckDuckGo search (plain text) |
| `beads_skill` | `cmd`, `id?`, `args?` | Wraps the `bd` CLI: list ready work, update/close tasks, prime AI context |

---

*Generated by AURA Scribe. Last updated from source: `aura_cli/server.py`,
`tools/aura_mcp_skills_server.py`, `tools/aura_control_mcp.py`,
`tools/agentic_loop_mcp.py`, `tools/github_copilot_mcp.py`,
`agents/skills/registry.py`, `.vscode/mcp.json`.*
