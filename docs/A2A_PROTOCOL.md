# AURA Agent-to-Agent (A2A) Protocol

## Overview

The Agent-to-Agent (A2A) protocol defines how AURA's internal agents communicate
with one another and with external peer agents.  AURA implements the open **A2A 1.0**
specification (see `core/a2a/`) on top of its existing FastAPI infrastructure.

### When agents communicate directly vs. through the orchestrator

| Scenario | Mechanism |
|---|---|
| Sequential pipeline phases (plan → critique → synthesize) | **Orchestrator-mediated** — `LoopOrchestrator` drives the cycle and passes payloads between agents in order |
| Cross-agent capability delegation (e.g. AURA delegating a review to a specialist peer) | **Direct A2A** — `A2AClient.delegate()` sends a task to a peer's `/a2a/tasks` endpoint |
| External system triggering AURA | **Webhook endpoints** — `/webhook/goal`, `/webhook/plan-review` via HTTP |
| Tool invocation (filesystem, shell, search) | **MCP endpoints** — `/execute` dispatches named tool calls |

AURA favours orchestrator-mediation for its own internal loop so that every
phase transition is logged, retried, and observable.  Direct A2A is used only
when a capability is owned by a *separate* agent process (e.g. a specialist
reviewer running on a different host).

---

## Message Format

Every A2A message follows a standard envelope regardless of transport layer.
Internal pipeline messages use Python dicts with identical fields; messages
sent over HTTP are JSON-encoded versions of the same structure.

```json
{
  "msg_id":     "550e8400-e29b-41d4-a716-446655440000",
  "from_agent": "planner",
  "to_agent":   "critic",
  "phase":      "plan",
  "payload":    { "...": "phase-specific content, see below" },
  "timestamp":  "2025-01-15T10:30:00Z"
}
```

### Field definitions

| Field | Type | Required | Description |
|---|---|---|---|
| `msg_id` | UUID string | yes | Unique identifier; used for deduplication and tracing |
| `from_agent` | string | yes | Logical name of the sending agent (e.g. `"planner"`, `"critic"`, `"synthesizer"`) |
| `to_agent` | string | yes | Logical name of the receiving agent; `"orchestrator"` for escalations |
| `phase` | string | yes | Pipeline phase tag — one of `plan`, `critique`, `synthesize`, `apply`, `test`, `reflect` |
| `payload` | object | yes | Phase-specific content (see Phase Handoffs below) |
| `timestamp` | ISO 8601 string | yes | UTC wall-clock time at message creation |

---

## Phase Handoffs

AURA's core loop progresses through six sequential phases.  The output of each
phase becomes the payload of the message delivered to the next agent.

### `plan` → `critique`

The **PlannerAgent** receives a goal from the orchestrator and returns a
structured plan.  Its output is wrapped and forwarded to **CriticAgent**.

**Planner output payload (inside `plan` phase message):**

```json
{
  "msg_id":     "a1b2c3d4-0000-0000-0000-000000000001",
  "from_agent": "planner",
  "to_agent":   "critic",
  "phase":      "plan",
  "payload": {
    "goal":  "Add pagination to the /tools endpoint",
    "steps": [
      "1. Add `page` and `page_size` query parameters to GET /tools",
      "2. Slice the tools list in the handler function",
      "3. Return `total`, `page`, `page_size`, and `items` in the response",
      "4. Update the OpenAPI snapshot with UPDATE_SNAPSHOTS=1"
    ],
    "estimated_complexity": "low",
    "memory_snapshot": "Previous pagination work: none found."
  },
  "timestamp":  "2025-01-15T10:30:01Z"
}
```

### `critique` → `synthesize`

The **CriticAgent** evaluates the plan for correctness, completeness, and
potential regressions.  Its output is forwarded to **SynthesizerAgent**.

**Critic output payload (inside `critique` phase message):**

```json
{
  "msg_id":     "a1b2c3d4-0000-0000-0000-000000000002",
  "from_agent": "critic",
  "to_agent":   "synthesizer",
  "phase":      "critique",
  "payload": {
    "approved": true,
    "score":    8,
    "issues": [
      {
        "severity":    "low",
        "description": "Step 4 should mention regenerating the CI snapshot, not just local",
        "suggestion":  "Add a note to run UPDATE_SNAPSHOTS=1 in CI before merging"
      }
    ],
    "reasoning": "Plan covers the happy path; pagination bounds checking not mentioned but low risk for internal endpoint.",
    "original_plan": ["...steps from planner..."]
  },
  "timestamp":  "2025-01-15T10:30:04Z"
}
```

### `synthesize` → `apply`

The **SynthesizerAgent** merges the plan and critique into a concrete task
bundle ready for code generation.

**Synthesizer output payload (inside `synthesize` phase message):**

```json
{
  "msg_id":     "a1b2c3d4-0000-0000-0000-000000000003",
  "from_agent": "synthesizer",
  "to_agent":   "coder",
  "phase":      "synthesize",
  "payload": {
    "tasks": [
      {
        "id":     "task_1",
        "title":  "Add pagination to GET /tools",
        "intent": "Add `page` and `page_size` query parameters to GET /tools\nSlice the tools list in the handler function\nReturn `total`, `page`, `page_size`, and `items` in the response\nUpdate the OpenAPI snapshot with UPDATE_SNAPSHOTS=1\n\nCritique:\nAdd a note to run UPDATE_SNAPSHOTS=1 in CI before merging",
        "files":  ["aura_cli/server.py", "tests/snapshots/openapi_schema.json"],
        "tests":  ["python3 -m pytest tests/test_openapi_contract.py -v"]
      }
    ]
  },
  "timestamp":  "2025-01-15T10:30:06Z"
}
```

---

## Error Protocol

### Agent-level errors

When an agent encounters an unrecoverable error it returns an error payload
instead of the normal output.  The orchestrator inspects the `"error"` key and
decides whether to retry or escalate.

```json
{
  "msg_id":     "a1b2c3d4-0000-0000-0000-000000000010",
  "from_agent": "critic",
  "to_agent":   "orchestrator",
  "phase":      "critique",
  "payload": {
    "error":   "LLM_TIMEOUT",
    "message": "Model did not respond within 60 s",
    "retryable": true,
    "attempt": 1
  },
  "timestamp":  "2025-01-15T10:31:00Z"
}
```

### Retry policy

| `error` code | Retryable | Max attempts | Back-off |
|---|---|---|---|
| `LLM_TIMEOUT` | yes | 3 | 2 s × attempt |
| `LLM_PARSE_ERROR` | yes | 2 | 1 s |
| `TOOL_NOT_FOUND` | no | — | — |
| `AUTH_DENIED` | no | — | — |
| `SCHEMA_INVALID` | yes | 1 | 0 s (re-prompt) |

### Escalation

If an agent exhausts its retry budget it transitions the task to `failed` and
sends a final escalation message to the orchestrator:

```json
{
  "from_agent": "critic",
  "to_agent":   "orchestrator",
  "phase":      "critique",
  "payload": {
    "error":     "MAX_RETRIES_EXCEEDED",
    "last_error": "LLM_TIMEOUT",
    "action":    "abort_cycle"
  }
}
```

The orchestrator logs the failure, stores a memory entry via `brain.remember()`,
and surfaces the error to the caller (e.g. the `/execute` or `/webhook/goal`
response).

### A2A task-level errors (HTTP)

When a peer agent task fails, the `A2ATask` state machine transitions to
`failed` and the HTTP response body contains:

```json
{
  "id":    "...",
  "state": "failed",
  "messages": [
    { "role": "agent", "content": "Failed: <reason>" }
  ]
}
```

Callers should check `state == "failed"` and inspect the last `agent` message.

---

## MCP Tool Invocation

Agents invoke external tools through the `/execute` endpoint, which acts as
an MCP (Model Context Protocol) gateway.

### Request format

```http
POST /execute
Authorization: Bearer <token>
Content-Type: application/json

{
  "tool_name": "run",
  "args": ["python3 -m pytest tests/ -q", "--timeout=120"]
}
```

### Named tools

| `tool_name` | Description | Key `args` |
|---|---|---|
| `run` | Execute a shell command in a sandboxed subprocess | `[command]` |
| `ask` | Prompt the active LLM model and return its text response | `[prompt]` |
| `env` | Return the safe subset of runtime environment variables | `[]` |
| `goal` | Trigger a full autonomous AURA loop for a goal | `[goal_string]` |

### Tool invocation inside an agent message

When an agent decides to call a tool, it embeds a `tool_call` block in its
payload.  The orchestrator extracts this and dispatches to `/execute`:

```json
{
  "msg_id":     "a1b2c3d4-0000-0000-0000-000000000020",
  "from_agent": "coder",
  "to_agent":   "orchestrator",
  "phase":      "apply",
  "payload": {
    "tool_call": {
      "tool_name": "run",
      "args":      ["python3 -m pytest tests/test_openapi_contract.py -v"]
    },
    "reason": "Verifying schema contract after endpoint change"
  },
  "timestamp":  "2025-01-15T10:32:00Z"
}
```

The orchestrator forwards the `tool_call` to `/execute`, collects the
streaming output, and returns the result in a new message to the agent.

### Streaming responses

Long-running `run` tool calls use Server-Sent Events (SSE).  Each event is a
JSON line:

```
data: {"type": "stdout", "line": "collected 42 items\n"}
data: {"type": "stdout", "line": "42 passed in 3.14s\n"}
data: {"type": "exit",   "code": 0}
```

Agents consuming SSE should buffer `stdout` lines and only act on `exit`.

---

## Examples

### Example 1 — Standard plan→critique→synthesize handoff

This sequence shows a complete internal loop for a simple goal.

**Step 1: Orchestrator → Planner**

```json
{
  "msg_id":     "ex1-0001",
  "from_agent": "orchestrator",
  "to_agent":   "planner",
  "phase":      "plan",
  "payload": {
    "goal":            "Fix the off-by-one error in the page_size clamp helper",
    "memory_snapshot": "Known weaknesses: boundary conditions in pagination helpers.",
    "similar_past_problems": "Previous fix: aura_cli/utils.py line 42, commit abc1234"
  },
  "timestamp": "2025-01-15T11:00:00Z"
}
```

**Step 2: Planner → Critic**

```json
{
  "msg_id":     "ex1-0002",
  "from_agent": "planner",
  "to_agent":   "critic",
  "phase":      "plan",
  "payload": {
    "goal":  "Fix the off-by-one error in the page_size clamp helper",
    "steps": [
      "1. Locate page_size clamp in aura_cli/utils.py",
      "2. Change `< max_size` to `<= max_size`",
      "3. Add unit test for boundary value max_size"
    ],
    "estimated_complexity": "trivial"
  },
  "timestamp": "2025-01-15T11:00:02Z"
}
```

**Step 3: Critic → Synthesizer**

```json
{
  "msg_id":     "ex1-0003",
  "from_agent": "critic",
  "to_agent":   "synthesizer",
  "phase":      "critique",
  "payload": {
    "approved": true,
    "score":    9,
    "issues":   [],
    "reasoning": "Fix is correct; boundary test in step 3 covers the regression."
  },
  "timestamp": "2025-01-15T11:00:05Z"
}
```

**Step 4: Synthesizer → Coder**

```json
{
  "msg_id":     "ex1-0004",
  "from_agent": "synthesizer",
  "to_agent":   "coder",
  "phase":      "synthesize",
  "payload": {
    "tasks": [
      {
        "id":     "task_1",
        "title":  "Fix off-by-one in page_size clamp",
        "intent": "Change `< max_size` to `<= max_size` in aura_cli/utils.py\nAdd unit test for boundary value max_size",
        "files":  ["aura_cli/utils.py", "tests/test_utils.py"],
        "tests":  ["python3 -m pytest tests/test_utils.py -v"]
      }
    ]
  },
  "timestamp": "2025-01-15T11:00:06Z"
}
```

---

### Example 2 — Cross-agent delegation via A2A HTTP

AURA delegates a specialised security review to an external `guardian` peer.

**Step 1: AURA A2AClient → Guardian POST /a2a/tasks**

```http
POST http://guardian-agent:8020/a2a/tasks
Content-Type: application/json

{
  "capability": "security_review",
  "message":    "Review the /execute endpoint for command-injection risks",
  "metadata": {
    "context_files": ["aura_cli/server.py"],
    "severity_threshold": "HIGH"
  }
}
```

**Step 2: Guardian responds (task accepted)**

```json
{
  "id":         "task-guardian-7f3a",
  "capability": "security_review",
  "state":      "working",
  "messages":   [{ "role": "user", "content": "Review the /execute endpoint..." }],
  "artifacts":  [],
  "created_at": 1736938210.5
}
```

**Step 3: AURA polls GET /a2a/tasks/task-guardian-7f3a**

```json
{
  "id":    "task-guardian-7f3a",
  "state": "completed",
  "messages": [
    { "role": "user",  "content": "Review the /execute endpoint..." },
    { "role": "agent", "content": "Denylist covers common injection vectors. Recommend adding path traversal check." }
  ],
  "artifacts": [
    {
      "name":      "security_findings",
      "mime_type": "application/json",
      "content": {
        "risk_level": "MEDIUM",
        "findings": [
          {
            "id":          "SEC-001",
            "description": "Path traversal via `..` in run command args",
            "severity":    "MEDIUM",
            "remediation": "Normalise arg[0] with Path.resolve() before execution"
          }
        ]
      }
    }
  ]
}
```

---

### Example 3 — Error recovery and retry

The planner LLM times out; the orchestrator retries once, then succeeds.

**Attempt 1 — Planner times out:**

```json
{
  "msg_id":     "ex3-0001",
  "from_agent": "planner",
  "to_agent":   "orchestrator",
  "phase":      "plan",
  "payload": {
    "error":     "LLM_TIMEOUT",
    "message":   "claude-3-sonnet did not respond within 60 s",
    "retryable": true,
    "attempt":   1
  },
  "timestamp": "2025-01-15T12:00:60Z"
}
```

**Orchestrator retries after 2 s back-off — Attempt 2 succeeds:**

```json
{
  "msg_id":     "ex3-0002",
  "from_agent": "planner",
  "to_agent":   "critic",
  "phase":      "plan",
  "payload": {
    "goal":  "Refactor memory store to use async I/O",
    "steps": [
      "1. Replace blocking sqlite3 calls in memory/brain.py with aiosqlite",
      "2. Update all callers to await the new async methods",
      "3. Run full test suite"
    ],
    "estimated_complexity": "medium",
    "_retry_attempt": 2
  },
  "timestamp": "2025-01-15T12:01:02Z"
}
```

---

## Related documents

- `docs/api/openapi.json` — Full OpenAPI 3.1 schema for all HTTP endpoints
- `core/a2a/` — Python implementation of A2A client, server, task, and agent card
- `tests/test_a2a.py` — Unit tests for the A2A implementation
- `docs/adr/ADR-001-api-server-decomposition.md` — Architecture decision record
- `docs/MCP_SERVERS.md` — MCP server configuration reference
