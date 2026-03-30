# Implementation Plan: Fleet Prompt Dispatcher

**Spec:** `2026-03-29-fleet-prompt-dispatcher-design.md`
**Date:** 2026-03-29
**Status:** Ready for implementation

---

## Phases Overview

| Phase | Deliverable | Depends on |
|-------|-------------|-----------|
| 0 | Prerequisites & env setup | — |
| 1 | WF-0: Master Dispatcher | Phase 0 |
| 2 | WF-1 through WF-5: Type handlers | Phase 1 |
| 3 | WF-6: Code Gen + PR Push | Phase 2 |
| 4 | End-to-end integration test | Phase 3 |
| 5 | GitHub Actions label trigger | Phase 3 |

---

## Phase 0 — Prerequisites & Environment Setup

**Goal:** Confirm all integration surfaces are reachable and configured before building workflows.

### Tasks

**P0-1: Verify AURA HTTP API endpoints**
- Confirm `POST http://localhost:8001/webhook/goal` accepts `{"goal","priority","metadata"}` and returns `{"goal_id","status"}`
- Confirm `GET http://localhost:8001/webhook/status/{goal_id}` returns `{"status","result"}` with terminal states `done`/`failed`
- Confirm `POST http://localhost:8001/execute` with `AGENT_API_ENABLE_RUN=1` accepts `{"tool_name":"run","args":[...]}`
- Document actual response shapes for each (will differ from n8n's expression expectations)

**P0-2: Verify AURA Skills Server**
- Confirm `POST http://localhost:8002/call` with `{"tool_name":"<skill>","args":{...}}` works for: `complexity_scorer`, `architecture_validator`, `security_scanner`, `doc_generator`, `code_clone_detector`, `error_pattern_matcher`, `generation_quality_checker`
- Note any skills that are slow (>10s) — these affect n8n timeout config

**P0-3: Configure environment variables**

On the **AURA host**:
```bash
export AGENT_API_TOKEN=<secret>
export AGENT_API_ENABLE_RUN=1
export AURA_REPO_PATH=/home/westonaaron675/aura-cli
```

On the **n8n host** (add as n8n credentials / env):
```bash
GITHUB_TOKEN=<pat with issues:write, pull_requests:write, contents:write>
AURA_API_TOKEN=<same value as AGENT_API_TOKEN above>
MCP_API_TOKEN=<skills server token>
AURA_API_URL=http://localhost:8001
AURA_SKILLS_URL=http://localhost:8002
AURA_FLEET_POLL_TIMEOUT=1200
AURA_REPO_PATH=/home/westonaaron675/aura-cli
```

**P0-4: Create GitHub labels**
```bash
gh label create "fleet:trigger"     --color "0075ca" --description "Trigger fleet dispatcher"
gh label create "fleet:in-progress" --color "e4e669" --description "Fleet run in progress"
gh label create "fleet:done"        --color "0e8a16" --description "Fleet run completed"
gh label create "fleet:blocked"     --color "d93f0b" --description "Fleet run blocked; needs human"
```

**P0-5: Configure GitHub webhook**
- Go to repo → Settings → Webhooks → Add webhook
- Payload URL: `<n8n-base>/webhook/fleet`
- Content type: `application/json`
- Events: **Issues** only
- Secret: store as `GITHUB_WEBHOOK_SECRET` on n8n host

---

## Phase 1 — WF-0: Master Dispatcher

**File:** `n8n-workflows/WF-0-master-dispatcher.json`

### Nodes to build (in order)

**Node 1 — Webhook Trigger**
- Type: `n8n-nodes-base.webhook`
- Path: `/fleet`
- Method: POST
- Response: Immediately (return 200 before processing)
- Authentication: HMAC-SHA256 signature check against `GITHUB_WEBHOOK_SECRET`

**Node 2 — Filter: skip terminal issues**
- Type: `n8n-nodes-base.if`
- Condition: `{{ $json.body.issue.labels }}` does NOT contain `fleet:done` AND does NOT contain `fleet:blocked`
- Also filter: `$json.body.action` is one of `opened`, `labeled`
- Also filter: `$json.body.issue.labels` contains `fleet:trigger`
- False branch → Stop and Respond (no-op)

**Node 3 — Extract issue fields**
- Type: `n8n-nodes-base.set`
- Fields to set:
  - `issue_number`: `{{ $json.body.issue.number }}`
  - `title`: `{{ $json.body.issue.title }}`
  - `body`: `{{ $json.body.issue.body }}`
  - `body_excerpt`: `{{ $json.body.issue.body.slice(0, 500) }}`
  - `repo_full_name`: `{{ $json.body.repository.full_name }}`

**Node 4 — LLM Classifier**
- Type: `n8n-nodes-base.openAi` (or Gemini equivalent)
- Model: `gpt-4o-mini` (cheap, fast) or `gemini-1.5-flash`
- System prompt: *(as specified in spec)*
- User message: `Issue title: {{title}}\nIssue body: {{body_excerpt}}`
- Output: set `issue_type` from classifier response (trim whitespace, lowercase)
- Fallback: if response not in `[bug, feature, refactor, security, docs]` → set `issue_type = "feature"`

**Node 5 — Label: fleet:in-progress**
- Type: `n8n-nodes-base.github`
- Operation: Add label to issue
- Label: `fleet:in-progress`
- Remove: `fleet:trigger` (optional, keeps label list clean)

**Node 6 — Switch**
- Type: `n8n-nodes-base.switch`
- Input: `{{ $json.issue_type }}`
- Cases: `bug` → WF-1, `feature` → WF-2, `refactor` → WF-3, `security` → WF-4, `docs` → WF-5
- Default: WF-2 (feature)

**Routing to sub-flows:** Use `n8n-nodes-base.executeWorkflow` nodes (one per case) to call WF-1 through WF-5, passing `issue_number`, `title`, `body`, `issue_type`, `repo_full_name`.

---

## Phase 2 — WF-1 through WF-5: Type Handlers

**Files:** `n8n-workflows/WF-{1..5}-*.json`

Each workflow follows the same structure — build WF-1 first and clone for the rest, changing only the skill list and goal prompt.

### Template structure

**Node 1 — Skills Fan-Out (parallel HTTP)**
- Type: `n8n-nodes-base.httpRequest` × N (one per skill, run in parallel via n8n parallel branches)
- URL: `{{ $env.AURA_SKILLS_URL }}/call`
- Method: POST
- Auth: Bearer `{{ $env.MCP_API_TOKEN }}`
- Timeout: 60s (skills can be slow on large repos)
- Body (per skill — see below)

**Skill args per handler:**

| Handler | Skill | Body |
|---------|-------|------|
| WF-1 | `complexity_scorer` | `{"tool_name":"complexity_scorer","args":{"project_root":"."}}` |
| WF-1 | `error_pattern_matcher` | `{"tool_name":"error_pattern_matcher","args":{"current_error":"{{body}}"}}` |
| WF-2 | `architecture_validator` | `{"tool_name":"architecture_validator","args":{"project_root":"."}}` |
| WF-2 | `dependency_analyzer` | `{"tool_name":"dependency_analyzer","args":{"project_root":"."}}` |
| WF-3 | `code_clone_detector` | `{"tool_name":"code_clone_detector","args":{"project_root":"."}}` |
| WF-3 | `refactoring_advisor` | `{"tool_name":"refactoring_advisor","args":{"project_root":"."}}` |
| WF-3 | `tech_debt_quantifier` | `{"tool_name":"tech_debt_quantifier","args":{"project_root":"."}}` |
| WF-4 | `security_scanner` | `{"tool_name":"security_scanner","args":{"project_root":"."}}` |
| WF-4 | `type_checker` | `{"tool_name":"type_checker","args":{"project_root":"."}}` |
| WF-5 | `doc_generator` | `{"tool_name":"doc_generator","args":{"project_root":"."}}` |
| WF-5 | `linter_enforcer` | `{"tool_name":"linter_enforcer","args":{"project_root":"."}}` |

**Node 2 — Merge skill outputs**
- Type: `n8n-nodes-base.merge` (mode: combine)
- Output: `skills_output` dict of `{ skill_name: result }`

**Node 3 — Build analyst_context**
- Type: `n8n-nodes-base.set`
- Compose `skills_output_summary` as a JSON string summary of top findings
- Pass through: `issue_number`, `issue_type`, `title`, `body`, `repo_full_name`, `skills_output`

**Node 4 — Execute WF-6**
- Type: `n8n-nodes-base.executeWorkflow`
- Workflow: WF-6
- Pass: full `analyst_context` payload

---

## Phase 3 — WF-6: Code Gen + PR Push

**File:** `n8n-workflows/WF-6-code-gen-pr-push.json`

This is the critical workflow. Build and test each node independently before chaining.

### Nodes

**Node 1 — Branch Create**
- GitHub API: `POST /repos/{repo}/git/refs`
- Body: `{"ref": "refs/heads/fleet/{{issue_type}}/{{issue_number}}", "sha": "<default branch HEAD SHA>"}`
- First: get HEAD SHA via `GET /repos/{repo}/git/ref/heads/main`

**Node 2 — Render goal prompt**
- Type: `n8n-nodes-base.set`
- Compose `goal_prompt`:
```
{{issue_type}} fix for issue #{{issue_number}}: {{title}}

Context from static analysis:
{{skills_output_summary}}

Original issue description:
{{body}}
```

**Node 3 — AURA goal dispatch**
- Type: `n8n-nodes-base.httpRequest`
- URL: `{{ $env.AURA_API_URL }}/webhook/goal`
- Method: POST
- Auth: Bearer `{{ $env.AURA_API_TOKEN }}`
- Body:
```json
{
  "goal": "{{goal_prompt}}",
  "priority": 5,
  "metadata": {
    "issue_number": "{{issue_number}}",
    "issue_type": "{{issue_type}}",
    "fleet_run": true
  }
}
```
- Extract: `goal_id` from response

**Node 4 — Poll goal status (loop)**
- Type: `n8n-nodes-base.wait` + `n8n-nodes-base.httpRequest` loop
- Poll: `GET {{ $env.AURA_API_URL }}/webhook/status/{{goal_id}}`
- Interval: 30s
- Max iterations: `AURA_FLEET_POLL_TIMEOUT / 30` (default 40)
- Terminal: status `done` → proceed; status `failed` → jump to escalation
- Timeout: jump to escalation

**Node 5 — Commit & Push**
- Type: `n8n-nodes-base.httpRequest`
- URL: `{{ $env.AURA_API_URL }}/execute`
- Method: POST
- Auth: Bearer `{{ $env.AURA_API_TOKEN }}`
- Body:
```json
{"tool_name": "run", "args": ["git -C {{AURA_REPO_PATH}} checkout fleet/{{issue_type}}/{{issue_number}} && git add -A && git commit -m 'fleet(#{{issue_number}}): {{title}}' && git push origin fleet/{{issue_type}}/{{issue_number}}"]}
```

**Node 6 — Read changed file for quality gate**
- Extract `applied_files[0]` from `result.history[-1].cycle_summary.applied_files`
- If empty → jump to escalation (score = 0)
- `POST /execute` with `{"tool_name":"run","args":["cat {{applied_files[0]}}"]}`
- Store as `generated_code`

**Node 7 — Quality Gate (with retry loop)**
- `POST {{ $env.AURA_SKILLS_URL }}/call`
- Body: `{"tool_name":"generation_quality_checker","args":{"task":"{{goal_prompt}}","generated_code":"{{generated_code}}"}}`
- Extract `quality_score`
- If score ≥ 70 → proceed to Node 8
- If score < 70 AND `quality_retry_count` < 2 → increment counter, back to Node 3 (re-dispatch with richer context)
- If score < 70 AND retries exhausted → jump to escalation

**Node 8 — PR Create**
- GitHub API: `POST /repos/{repo}/pulls`
- Body:
```json
{
  "title": "fleet(#{{issue_number}}): {{title}}",
  "head": "fleet/{{issue_type}}/{{issue_number}}",
  "base": "main",
  "body": "Closes #{{issue_number}}\n\n**Fleet dispatcher automated PR**\n\nIssue type: {{issue_type}}\nQuality score: {{quality_score}}/100\n\n### Analyst context\n{{skills_output_summary}}"
}
```
- Extract: `pr_number`

**Node 9 — Wait for CI (poll loop)**
- `GET /repos/{repo}/commits/{head_sha}/check-runs`
- Poll 30s intervals, max 15 min
- All `status: completed` AND all `conclusion: success` → proceed
- Any `conclusion: failure` AND `ci_retry_count` < 3 → trigger `/fix` (or re-run failed checks), increment counter
- CI retries exhausted → jump to escalation

**Node 10 — Auto-merge**
- GitHub API: `PUT /repos/{repo}/pulls/{pr_number}/merge`
- Merge method: `squash`

**Node 11 — Label: fleet:done + close issue**
- Add label `fleet:done`, remove `fleet:in-progress`
- Close issue: `PATCH /repos/{repo}/issues/{issue_number}` with `{"state":"closed"}`

**Node 12 — Escalation (reachable from any failure node)**
- Add label `fleet:blocked`, remove `fleet:in-progress`
- Post comment (structured blocker comment per spec)
- Stop

---

## Phase 4 — End-to-End Integration Test

### Test sequence

1. **Smoke test (dry run):** Create a test issue in a fork with `fleet:trigger`, verify WF-0 fires, classifies, routes to WF-N, and calls AURA Skills — stop before WF-6 by pointing WF-6 at a mock endpoint.

2. **WF-6 isolated test:** Manually invoke WF-6 with a pre-built `analyst_context` payload. Verify:
   - Branch created
   - AURA goal dispatched and polled to completion
   - Files committed and pushed
   - Quality gate fires with real score
   - PR created with correct title and body

3. **Quality gate failure test:** Inject a bad goal (e.g., "do nothing") — verify score < 70, quality retries fire, escalation triggers after 2 retries.

4. **CI failure test:** Create a PR that introduces a linting error — verify CI fail detection, up to 3 retries, then `fleet:blocked` label.

5. **Full happy path:** Real issue in the target repo → `fleet:trigger` label → full pipeline → PR merged → `fleet:done` → issue closed.

---

## Phase 5 — GitHub Actions Label Trigger (optional hardening)

Add a GitHub Actions workflow that fires when `fleet:trigger` is applied and POSTs to the n8n webhook — this provides a reliable trigger fallback if the repo webhook is flaky.

**File:** `.github/workflows/fleet-trigger.yml`

```yaml
name: Fleet Trigger
on:
  issues:
    types: [labeled]
jobs:
  dispatch:
    if: github.event.label.name == 'fleet:trigger'
    runs-on: ubuntu-latest
    steps:
      - name: Dispatch to n8n
        run: |
          curl -X POST "${{ secrets.N8N_FLEET_WEBHOOK_URL }}" \
            -H "Content-Type: application/json" \
            -d '${{ toJson(github.event) }}'
```

Required secret: `N8N_FLEET_WEBHOOK_URL`

---

## File Layout

```
n8n-workflows/
├── WF-0-master-dispatcher.json
├── WF-1-bug-fix-handler.json
├── WF-2-feature-handler.json
├── WF-3-refactor-handler.json
├── WF-4-security-handler.json
├── WF-5-docs-handler.json
└── WF-6-code-gen-pr-push.json

.github/workflows/
└── fleet-trigger.yml          ← Phase 5

docs/superpowers/specs/
├── 2026-03-29-fleet-prompt-dispatcher-design.md  ← approved spec
└── 2026-03-29-fleet-prompt-dispatcher-plan.md    ← this file
```

---

## Implementation Order

Build in this order to enable incremental testing:

1. Phase 0 (env + labels + webhook) — unblocks everything
2. WF-6 in isolation (mock inputs) — the hardest workflow; validate first
3. WF-0 (dispatcher) — needs nothing from WF-1..5
4. WF-1 (bug handler) — validate skill fan-out pattern
5. WF-2..5 (clone from WF-1, change skills + prompt)
6. Wire WF-0 → WF-1..5 → WF-6
7. End-to-end integration tests
8. fleet-trigger.yml (optional hardening)
