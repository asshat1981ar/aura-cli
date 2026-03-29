# Design Spec: Fleet Prompt Dispatcher

**Date:** 2026-03-29
**Status:** Draft
**Scope:** n8n workflows (7 new), GitHub Issues, AURA CLI API, `agents/skills/`, GitHub Actions
**Goal:** Fully autonomous GitHub Issue → code change → PR → merge loop, dispatched via n8n, evaluated by AURA Skills, and executed by AURA Loop.

---

## Problem Statement

AURA CLI can generate code changes autonomously, but has no self-directed intake channel for production work. GitHub Issues are the natural source of engineering work, but there is no automated path from "issue filed" to "branch created, code written, tests passing, PR merged."

The fleet prompt dispatcher closes this gap: a GitHub Issue with a `fleet:trigger` label kicks off a fully autonomous pipeline that classifies the issue, runs targeted analysis, generates code, creates a PR, and auto-merges on success — with structured escalation when it cannot proceed.

---

## Architecture

### Trigger

- GitHub Issues (and PRs) with the `fleet:trigger` label
- n8n webhook receives GitHub `issues` event via a configured webhook
- Only `opened` and `labeled` actions are processed
- Issues already labeled `fleet:done` or `fleet:blocked` are skipped

### Top-Level Flow

```
GitHub Issue (fleet:trigger label)
        │
        ▼
WF-0: Master Dispatcher
  ├─ Webhook receiver
  ├─ LLM classifier (issue type → bug / feature / refactor / security / docs)
  └─ Switch node → 5 sub-flows
        │
        ├─► WF-1: Bug-Fix Handler
        ├─► WF-2: Feature Handler
        ├─► WF-3: Refactor Handler
        ├─► WF-4: Security Handler
        └─► WF-5: Docs Handler
              │
              ▼
          (type-specific AURA Skills fan-out)
              │
              ▼
          WF-6: Code Gen + PR Push
```

### Workflow Inventory

| ID | Name | Purpose |
|----|------|---------|
| WF-0 | Master Dispatcher | Webhook intake, LLM classification, Switch routing |
| WF-1 | Bug-Fix Handler | complexity_scorer + error_pattern_matcher skills, bug-focused prompt |
| WF-2 | Feature Handler | architecture_validator + dependency_analyzer skills, feature prompt |
| WF-3 | Refactor Handler | code_clone_detector + refactoring_advisor + tech_debt_quantifier skills |
| WF-4 | Security Handler | security_scanner + type_checker skills, security-focused prompt |
| WF-5 | Docs Handler | doc_generator + linter_enforcer skills, docs-focused prompt |
| WF-6 | Code Gen + PR Push | AURA Loop goal dispatch, git commit/push, quality gate, PR creation, auto-merge |

---

## Detailed Phase Design

### WF-0: Master Dispatcher

**Nodes:**
1. **Webhook Trigger** — receives `POST /webhook/fleet` from GitHub
2. **Filter** — skip if labeled `fleet:done`, `fleet:blocked`; accept `fleet:trigger`
3. **LLM Classifier** (OpenAI/Gemini) — prompt: classify issue as `bug`, `feature`, `refactor`, `security`, or `docs` based on title + body
4. **Switch** — routes on classifier output to WF-1 through WF-5
5. **Label: fleet:in-progress** — applied to issue before sub-flow dispatch

**LLM Classifier Prompt Template:**
```
You are an issue classifier for a Python CLI project.
Classify the following GitHub issue into exactly one category:
  bug, feature, refactor, security, docs

Issue title: {{title}}
Issue body (first 500 chars): {{body_excerpt}}

Respond with a single word from the list above.
```

### WF-1 through WF-5: Type-Specific Handlers

Each handler runs a **AURA Skills fan-out** via parallel HTTP calls to the AURA Skills Server (`POST http://localhost:8002/call`) then packages results as `analyst_context` for WF-6.

**Bug-Fix (WF-1):** `complexity_scorer` + `error_pattern_matcher`
- `complexity_scorer`: `{"project_root": "."}`
- `error_pattern_matcher`: `{"current_error": "{{issue_body}}"}` — most effective when the issue body contains a traceback; otherwise returns low-confidence match (acceptable for routing context)

**Feature (WF-2):** `architecture_validator` + `dependency_analyzer`
- `architecture_validator`: `{"project_root": "."}`
- `dependency_analyzer`: `{"project_root": "."}`

**Refactor (WF-3):** `code_clone_detector` + `refactoring_advisor` + `tech_debt_quantifier`
- All three: `{"project_root": "."}`

**Security (WF-4):** `security_scanner` + `type_checker`
- `security_scanner`: `{"project_root": "."}`
- `type_checker`: `{"project_root": "."}`

**Docs (WF-5):** `doc_generator` + `linter_enforcer`
- `doc_generator`: `{"project_root": "."}`
- `linter_enforcer`: `{"project_root": "."}`

**Analyst context payload passed to WF-6:**
```json
{
  "issue_number": 42,
  "issue_type": "bug",
  "title": "...",
  "body": "...",
  "skills_output": { "<skill_name>": { ... } }
}
```

### WF-6: Code Gen + PR Push

**Nodes:**
1. **Branch Create** — creates `fleet/{type}/{issue-number}` via GitHub API
2. **AURA Loop dispatch** — `POST http://localhost:8001/webhook/goal`
   ```json
   {
     "goal": "<rendered goal prompt>",
     "priority": 5,
     "metadata": {
       "issue_number": "{{issue_number}}",
       "issue_type":   "{{issue_type}}",
       "fleet_run":    true
     }
   }
   ```
   Response: `{"goal_id": "<id>", "status": "queued"}`. Store `goal_id` for step 3.
   Auth: `Authorization: Bearer {{AURA_API_TOKEN}}` header required.
3. **Poll goal status** — `GET http://localhost:8001/webhook/status/{{goal_id}}`
   Poll at 30s intervals, max 20 min (configurable via `AURA_FLEET_POLL_TIMEOUT`).
   Terminal states:
   - `"status": "done"` → proceed; capture `result` payload (includes `cycle_summary`)
   - `"status": "failed"` → treat as AURA run failure; apply retry budget
   - `"status": "queued"` / `"running"` → continue polling
4. **Commit & Push** — after AURA run completes, commit the changes to the fleet branch:
   ```
   POST http://localhost:8001/execute
   {"tool_name": "run", "args": ["git -C /repo checkout fleet/{{issue_type}}/{{issue_number}} && git add -A && git commit -m 'fleet(#{{issue_number}}): {{title}}' && git push origin fleet/{{issue_type}}/{{issue_number}}"]}
   ```
   This step requires `AGENT_API_ENABLE_RUN=1` on the AURA server. Scope this env flag to the AURA host only; do not set it in the n8n environment.
5. **Quality Gate** — calls `generation_quality_checker` skill; score must be ≥ 70
6. **PR Create** — creates PR with structured description linking to issue
7. **Wait for CI** — polls PR check status (30s intervals, max 15 min)
8. **Auto-merge** — merges PR on CI green
9. **Label: fleet:done** — applied to issue; issue closed

**Goal prompt injected into AURA loop dispatch:**
```
{{issue_type}} fix for issue #{{issue_number}}: {{title}}

Context from static analysis:
{{skills_output_summary}}

Original issue description:
{{body}}
```

> **Note:** `webhook/goal` uses an in-memory queue on the AURA server. An AURA server restart between goal submission and polling will cause a 404 on the status endpoint and trigger AURA run failure escalation. Keep the AURA server alive for the full duration of a fleet run.

---

## Quality Gate

The `generation_quality_checker` skill scores generated code on a 0–100 scale against:
- Intent match (does the change address the issue)
- Test presence
- Structural validity

**Threshold:** score ≥ 70 required before PR push.

**Quality gate input:** extract `generated_code` from the AURA goal result returned at step 3:
```json
{
  "tool_name": "generation_quality_checker",
  "args": {
    "task": "{{rendered goal prompt}}",
    "generated_code": "{{result.cycle_summary.generated_code}}"
  }
}
```
If `generated_code` is absent from the AURA result (no code block produced), treat as score = 0 and escalate immediately without consuming a quality retry.

**Quality retry:** if score < 70, re-run AURA loop dispatch with richer context (include full skill output detail). Max **2 quality retries** before escalation.

---

## Retry Budget

| Failure type | Max retries | Action on exhaustion |
|---|---|---|
| Quality gate fails (score < 70) | 2 | Escalate |
| CI checks fail post-PR | 3 | Escalate |
| AURA goal run timeout / failed status | 1 | Escalate |
| Skills API unreachable | 0 | Escalate immediately |

---

## Escalation

When retry budget is exhausted:

1. Apply `fleet:blocked` label to the issue
2. Remove `fleet:in-progress` label
3. Post structured blocker comment:

```markdown
## 🚫 Fleet Dispatcher Blocked

**Issue type:** {{issue_type}}
**Failure reason:** {{reason}} (after {{retry_count}} retries)
**Last quality score:** {{quality_score}}/100 (threshold: 70)
**Last CI status:** {{ci_status}}

**Analyst context available:** [skill outputs attached]

Human intervention required.
```

---

## PR Naming

```
fleet/{type}/{issue-number}
```

Examples: `fleet/bug/142`, `fleet/feature/89`, `fleet/security/201`

PR title format: `fleet(#{issue}): {issue_title}`

---

## Data Flow

```
GitHub Webhook → WF-0 (classify) → WF-N (skills fan-out)
    → WF-6 (AURA /run) → quality gate → PR create
    → CI wait → auto-merge → fleet:done
```

All intermediate state (analyst context, quality scores, retry counts) is passed as n8n expression data between nodes. No external state store required for the happy path.

---

## Error Handling

- All HTTP calls to AURA use n8n's built-in retry node (3 retries, 5s backoff)
- Webhook receiver returns 200 immediately and processes asynchronously
- Unrecognized issue types default to `feature` classification
- If the LLM classifier fails, fallback to `feature`
- `fleet:in-progress` label is always cleaned up on terminal outcomes (done or blocked)

---

## Integration Points

| Component | Interface |
|-----------|-----------|
| GitHub Issues | GitHub webhook + GitHub REST API (`gh` token) |
| n8n | Workflow webhook triggers; n8n internal chaining |
| AURA Skills Server | `POST http://localhost:8002/call` |
| AURA Loop API | `POST http://localhost:8001/webhook/goal` (auth via `AGENT_API_TOKEN` on AURA server) |
| AURA Goal Status | `GET http://localhost:8001/webhook/status/{goal_id}` |
| AURA Commit/Push | `POST http://localhost:8001/execute` (requires `AGENT_API_ENABLE_RUN=1` on AURA server) |
| generation_quality_checker | AURA Skills Server via skill name |

---

## Configuration

Environment variables required on the n8n host:

| Variable | Description |
|----------|-------------|
| `GITHUB_TOKEN` | PAT with `issues:write`, `pull_requests:write`, `contents:write` |
| `AURA_API_TOKEN` | Bearer token for AURA HTTP API; must match `AGENT_API_TOKEN` on the AURA server |
| `MCP_API_TOKEN` | Token for AURA Skills Server |
| `AURA_API_URL` | Default: `http://localhost:8001` |
| `AURA_SKILLS_URL` | Default: `http://localhost:8002` |
| `AURA_FLEET_POLL_TIMEOUT` | Goal status poll timeout in seconds (default: 1200 / 20 min) |

---

## Testing

- **Unit:** mock AURA `/run` and Skills API; assert correct skill fan-out per issue type
- **Integration:** create a test issue with `fleet:trigger`, verify full flow against a sandbox repo
- **Quality gate test:** inject a low-quality code response; verify escalation fires before PR push
- **Retry test:** simulate CI failure 3 times; verify `fleet:blocked` is applied

---

## Out of Scope

- Multi-repo fleet dispatch (single repo only)
- Issue de-duplication / conflict detection between parallel fleet runs
- Cost tracking for LLM classifier calls
- Async status dashboard (fleet run history)
