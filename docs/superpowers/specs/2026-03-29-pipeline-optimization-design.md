# Design Spec: Pipeline Optimization for AURA Dev Suite + AURA CLI

**Date:** 2026-03-29
**Status:** Reviewed (9 issues fixed)
**Scope:** core/orchestrator.py, aura_cli/server.py, agents/skills/*, n8n workflows (5 new + 2 updated), aura.config.json
**Primary Optimization Target:** Autonomy + Quality — close the n8n ↔ AURA loop so complex goals are reviewed by multi-agent Dev Suite before being applied, and all outcomes flow back as lessons without human touch.

---

## Problem Statement

AURA CLI and the n8n Aura Dev Suite are capable but **disconnected**:

| System | Strength | Current Gap |
|--------|----------|-------------|
| AURA CLI orchestrator (10 phases) | Deep automated code changes | Operates in isolation; n8n only notified after the fact |
| n8n Aura Dev Suite (65 nodes, 7 agents) | Wide multi-agent review, GitHub integration | Cannot trigger AURA phases or inject into mid-cycle |
| n8n Parallel Skills Runner | 4 parallel analyses + eval_optimizer | Webhook-triggered but no loop; single-shot |
| n8n Goal Orchestrator | Complexity classification + routing | manualTrigger only; not wired to real AURA |

**Root causes of inefficiency:**
1. No closed feedback loop — AURA reflects but lessons don't re-enter n8n
2. No pre-act quality gate — code is applied before multi-agent review
3. No goal intelligence — all goals take the same path regardless of complexity
4. The 65-node Dev Suite agents (Code Architect, Reviewer, DevOps, etc.) are disconnected from the CLI loop
5. Failed executions (e.g., exec 131) go unretried; no circuit breaker or retry policy
6. No pipeline observability — no unified trace linking n8n execution IDs to AURA cycle IDs

---

## Goals

1. **Autonomous end-to-end loop**: GitHub PR / CLI goal → classify → Dev Suite review → AURA act → verify → report back → no human touch needed
2. **Pre-act quality gate**: n8n Dev Suite reviews AURA's _plan_ before code is written, injecting critique from 7 specialist agents
3. **Closed feedback loop**: AURA reflect output → n8n lesson store → improves future plan/critique prompts
4. **Goal intelligence**: Route simple goals to AURA-only fast lane; complex/risky to AURA + Dev Suite dual lane
5. **Pipeline observability**: Every cycle tagged with a `pipeline_run_id` traceable across n8n executions + AURA phases
6. **Retry + circuit breaker**: Failed n8n executions auto-retry with backoff; circuit opens after 3 failures

---

## Approaches Considered

### Approach A — Unified Event Bus
n8n becomes a central Kafka-style event router. AURA phases emit events to n8n webhooks; n8n routes, aggregates, and triggers follow-on workflows.

| Pros | Cons |
|------|------|
| Fully loose coupling | n8n becomes critical path for every AURA phase |
| Easy to monitor in n8n canvas | HTTP overhead on every phase transition |
| Any external system can subscribe | Complex fan-in/fan-out for 10 phases |

### Approach B — AURA-Native Plugin Hooks
Extend AURA's orchestrator with native `before_phase` / `after_phase` hooks. Dev Suite agents become AURA phase plugins called inline.

| Pros | Cons |
|------|------|
| Single control plane (AURA owns everything) | Tight coupling; breaks if n8n is down |
| Lowest latency (no webhook round-trips) | Orchestrator complexity grows significantly |

### Approach C — Parallel Lanes with Smart Merge ✅ **RECOMMENDED**
Two parallel tracks triggered by a **Goal Intelligence Router**:
- **Fast Lane** (simple goals): AURA CLI alone → notify
- **Dual Lane** (complex/risky goals): AURA CLI + Dev Suite in parallel → merge → apply

A new **Pipeline Coordinator** workflow orchestrates lanes, retries, and the feedback loop. This is the recommended approach because:
- It avoids putting n8n on the AURA critical path for simple goals
- It brings the Dev Suite's 7 specialist agents in only where they add value
- It keeps both systems independently operable if one is down
- It maps directly to the existing Goal Orchestrator complexity classification already in n8n

---

## Recommended Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GOAL INTELLIGENCE ROUTER (n8n)                    │
│   GitHub PR webhook / CLI goal / scheduled scan                      │
│         │                                                             │
│   ┌─────▼──────┐                                                      │
│   │  Classify   │  complexity: low / medium / high                    │
│   │  + Route    │  risk: low / medium / high                          │
│   └──┬─────┬───┘                                                      │
│      │     │                                                           │
│   FAST     DUAL LANE                                                  │
│   LANE     │                                                           │
│      │   ┌─┴────────────────────────────────┐                        │
│      │   │  PARALLEL ANALYSIS + REVIEW       │                        │
│      │   │                                   │                        │
│      │   │  ┌─────────────────┐  ┌────────┐ │                        │
│      │   │  │ AURA Dev Suite  │  │ Skills │ │                        │
│      │   │  │ 7 Agent Review  │  │ Runner │ │                        │
│      │   │  │ (plan quality   │  │ (4     │ │                        │
│      │   │  │  gate)          │  │ skills │ │                        │
│      │   │  └────────┬────────┘  └───┬────┘ │                        │
│      │   │           │               │       │                        │
│      │   │  ┌────────▼───────────────▼────┐ │                        │
│      │   │  │     Merge + Eval Optimizer   │ │                        │
│      │   │  │   (score ≥ 0.75 to proceed)  │ │                        │
│      │   │  └────────────────┬────────────┘ │                        │
│      │   └───────────────────┼──────────────┘                        │
│      │                       │                                        │
│      └──────────┬────────────┘                                        │
│                 │                                                      │
│         ┌───────▼────────┐                                            │
│         │  AURA CLI ACT  │  POST /webhook/goal → cycles              │
│         │  (apply phase) │                                            │
│         └───────┬────────┘                                            │
│                 │                                                      │
│         ┌───────▼────────────────────────────┐                       │
│         │  POST-CYCLE FEEDBACK LOOP           │                       │
│         │  reflect output → Notification Hub  │                       │
│         │  → Lesson Store → future plan ctx   │                       │
│         └────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Inventory (5 new + 2 updated)

### New Pipelines

| # | Pipeline Name | Trigger | Purpose |
|---|--------------|---------|---------|
| P1 | **Goal Intelligence Router** | Webhook, GitHub PR event, schedule | Classify goal, select lane, dispatch to P2 or P4 |
| P2 | **Dev Suite Quality Gate** | Called by P1 (dual lane) | Run Dev Suite 7-agent review on AURA's plan; return critique + approve/block decision |
| P3 | **Pipeline Coordinator** | Called by P1 | Orchestrate parallel lanes, merge results, trigger AURA act, retry logic, circuit breaker |
| P4 | **Feedback Loop** | Called after AURA reflect | Store cycle lessons, update skill weights in n8n memory, log pipeline run metrics |
| P5 | **Observability Collector** | Polling + webhook | Aggregate execution IDs, phase timings, scores; write to pipeline_runs table |

### Updated Pipelines

| # | Pipeline Name | Change |
|---|--------------|--------|
| U1 | **AURA Parallel Skills Runner** | Add retry logic; wire Eval Optimizer error fallback; fix execution 131 crash |
| U2 | **AURA Goal Orchestrator** | Replace manualTrigger with webhookTrigger; wire to Pipeline Coordinator (P3) |

---

## P1 — Goal Intelligence Router

**Nodes:**
```
Webhook (POST /webhook/goal-route)
  → Set Goal Metadata (id, timestamp, source)
  → Classify Complexity (HTTP → POST /execute body: {"tool_name": "ask", "args": [classify_prompt]})
  → Extract Fields (complexity, risk, goal_type)
  → Route by Lane (Switch node)
      → [fast lane path]
          → AURA CLI Direct (HTTP → POST /webhook/goal)
          → Poll Status (loop: GET /webhook/status/{goal_id})
          → Format Result → Respond to Webhook
      → [dual lane path]
          → Pipeline Coordinator (P3) sub-workflow call (P3 owns polling + AURA trigger)
          → Receive P3 result → Format Result → Respond to Webhook
```

**Classification prompt** (sent to AURA `/execute`):
```
Classify this goal: "{goal}"
Output JSON: {
  "complexity": "low|medium|high",
  "risk": "low|medium|high",
  "goal_type": "refactor|feature|bugfix|security|docs|test",
  "reasoning": "..."
}
```

**Lane routing rules:**

| complexity | risk | Lane |
|------------|------|------|
| low | any | Fast |
| medium | low | Fast |
| medium | medium-high | Dual |
| high | any | Dual |

---

## P2 — Dev Suite Quality Gate

**Purpose:** Run the 65-node Aura Dev Suite against AURA's generated _plan_ (not yet applied code), get multi-agent critique, and return an approve/block signal.

**Integration point:** P2 is an **external pre-validation step** called by P3 (n8n-side) _before_ AURA is triggered. P3 sends the goal description plus any available context to P2's webhook; P2 returns `{approved: bool, critique: [...], suggestions: [...]}`. The AURA-side `before_act` hook (see AURA CLI Changes) is a _separate, optional_ enhancement that injects P2's critique into the act phase context after the fact — it does not gate execution.

**New AURA server endpoint needed:** `POST /webhook/plan-review` — accepts task_bundle, returns plan text for Dev Suite to review.

**Nodes:**
```
Webhook (POST /webhook/plan-review)
  → Extract Plan Text
  → Parallel fan-out:
      ├── Code Architect Agent (review plan structure)
      ├── Security Auditor Agent (flag risky patterns)
      └── Test Engineer Agent (identify missing test coverage)
  → Review Council Chair (synthesize votes → approve/needs-revision/block)
  → Format Gate Decision {approved, critique, suggestions, confidence}
  → Respond to Webhook
```

**AURA orchestrator hook (optional enrichment):** New `enrich_act_context()` call in `core/orchestrator.py` — reads `n8n_connector.quality_gate_critique` from the cycle context dict (placed there by P3 via the webhook goal payload's `metadata.critique` field). Does **not** make any blocking HTTP call; critique is pre-fetched by P3 and injected into `/webhook/goal` payload.

---

## P3 — Pipeline Coordinator

**Purpose:** Orchestrate the dual lane — run P2 (Dev Suite review) and Skills Runner in parallel, merge via Eval Optimizer, then trigger AURA act only if score ≥ 0.75.

**Nodes:**
```
Sub-Workflow Trigger (called by P1)
  → Set Run ID (pipeline_run_id = uuid)
  → Parallel fan-out:
      ├── P2 Dev Suite Quality Gate (HTTP request to webhook)
      └── Skills Runner (execute_workflow: 81vcXiXMfSprsc8t)
  → Wait for Both (Merge node, all inputs)
  → Eval Optimizer (HTTP → POST 172.18.0.1:8002/call, tool_name: eval_optimizer)
  → Check Score (IF score >= 0.75)
      → [pass] Trigger AURA Act (POST /webhook/goal with plan + critique)
      → [fail] Loop back to P2 with critique (max 2 re-reviews)
  → Poll AURA Status (GET /webhook/status/{goal_id}, loop until done/failed)
  → Emit to P4 Feedback Loop
  → Emit to P5 Observability
  → Return Result
```

**Circuit breaker state** (stored in n8n static data):
```json
{
  "failures": 0,
  "last_failure": null,
  "state": "closed|open|half-open",
  "open_until": null
}
```
Circuit opens after 3 consecutive AURA failures; resets after 5-minute cooldown.

---

## P4 — Feedback Loop

**Purpose:** After every AURA cycle completes, harvest the reflect output and feed lessons back into n8n memory nodes so future Dev Suite agents have richer context.

**Trigger:** Called by P3 after AURA respond; also called directly by `agents/reflector.py` via `_notify_n8n()`.

**Nodes:**
```
Webhook (POST /webhook/aura-reflect)
  → Extract Cycle Summary (goal, outcome, lessons)
      Note: skill_scores passed only if orchestrator includes phase_outputs["skill_dispatch"] in webhook payload
  → Store Lesson (Memory node — window: 50 items, append mode)
  → Update Skill Weights (HTTP → POST /execute body: {"tool_name": "run", "args": ["update_skill_weights"]})
  → Check Pattern (IF similar failures > 2 → trigger evolution)
  → Conditional: Trigger Evolution Workflow (evolution_skill via /call)
  → Respond OK
```

**Memory node** feeds into Dev Suite agents' system prompts: "Past lessons from similar goals: {memory}".

---

## P5 — Observability Collector

**Purpose:** Unified trace of every pipeline run — correlate n8n execution IDs with AURA cycle IDs, phase timings, skill scores, and gate decisions.

**Data model (n8n static data table `pipeline_runs`):**
```json
{
  "pipeline_run_id": "uuid",
  "goal": "...",
  "lane": "fast|dual",
  "started_at": "ISO8601",
  "aura_goal_id": "...",
  "n8n_execution_ids": ["131", "132"],
  "phases": {
    "classify": {"duration_ms": 200, "result": "high/high"},
    "dev_suite_gate": {"duration_ms": 8000, "approved": true, "score": 0.82},
    "skills_runner": {"duration_ms": 90000, "eval_score": 0.825},
    "aura_act": {"duration_ms": 45000, "status": "success"}
  },
  "final_status": "success|failed|blocked",
  "lessons_stored": 3
}
```

---

## AURA CLI Changes

### 1. `core/orchestrator.py` — `before_act` quality gate hook

```python
# After synthesize phase, before act phase:
if self._should_run_quality_gate(goal, task_bundle):
    gate_result = await self._call_quality_gate(task_bundle)
    if not gate_result.get("approved"):
        # Inject critique into act context
        task_bundle["quality_gate_critique"] = gate_result.get("critique", [])
        task_bundle["quality_gate_suggestions"] = gate_result.get("suggestions", [])

def _should_run_quality_gate(self, goal: str, task_bundle: dict) -> bool:
    cfg = self.config.get("n8n_connector", {})
    return (
        cfg.get("quality_gate_enabled", False)
        and cfg.get("goal_complexity", "low") in ("medium", "high")
    )
```

### 2. `aura_cli/server.py` — New endpoint + WebhookGoalRequest change

```python
POST /webhook/plan-review
  # Accept: {task_bundle, goal, pipeline_run_id}
  # Return: plan text formatted for Dev Suite review
  # Auth: require_auth

# Also extend WebhookGoalRequest:
class WebhookGoalRequest(BaseModel):
    goal: str
    metadata: dict = {}  # NEW — carries: {"complexity": "high", "pipeline_run_id": "uuid", "critique": [...]}
                         # orchestrator reads metadata["complexity"] for quality gate routing
```

### 3. `agents/reflector.py` — Extended `_notify_n8n()`

```python
# Current: POSTs to notification_webhook
# Change: POST to /webhook/aura-reflect (P4 feedback loop)
# Include: lessons, pipeline_run_id (if set in cycle context)
    # Note: skill_scores require phase_outputs['skill_dispatch'] to be passed into reflect — add as prerequisite
```

### 4. `aura.config.json` — New keys

```json
"n8n_connector": {
  ...existing...,
  "quality_gate_enabled": false,
  "quality_gate_webhook": "http://localhost:5678/webhook/plan-review",
  "feedback_loop_webhook": "http://localhost:5678/webhook/aura-reflect",
  "pipeline_coordinator_webhook": "http://localhost:5678/webhook/pipeline-coordinator",
  "circuit_breaker_enabled": true
}
```

---

## Error Handling

| Scenario | Handler |
|----------|---------|
| P2 Dev Suite times out (>60s) | P3 bypasses gate, logs warning, proceeds with AURA act |
| Skills Runner fails (exec 131 crash) | P3 retries once with simplified payload; if still failing, proceeds without skill scores |
| AURA act fails (verify error) | P3 signals failure to P4; P4 stores lesson; circuit-breaker increments failure count |
| Eval Optimizer score < 0.75 after 2 re-reviews | P3 blocks goal, notifies Notification Hub with full critique |
| n8n unreachable from AURA | `_notify_n8n()` catches exception, logs, does not block cycle |
| **P1 classification fails** (timeout / invalid JSON from /execute) | Default to fast lane, log warning — goal is never dropped |
| **P3 unreachable from P1** (crashed/down) | P1 falls back to fast lane (direct AURA), logs error, alerts Notification Hub on repeated failures |
| **AURA webhook auth failure** (401 from /webhook/goal) | Retry once after 5s; if still 401, alert Notification Hub and halt until credentials rotated |
| Circuit breaker open | P1 fast-lane all goals; skip dual lane until half-open |

---

## Implementation Phases

### Phase 1 — Foundation (unblocks all other phases)
- [ ] Create P3 stub workflow (webhook that accepts dual-lane request, returns `{status: 'queued', run_id: uuid}` placeholder until Phase 3 replaces it)
- [ ] Fix P1 basis: update Goal Orchestrator to webhookTrigger (U2)
- [ ] Fix Eval Optimizer node crash in exec 131 (U1)
- [ ] Add `POST /webhook/plan-review` to `aura_cli/server.py`
- [ ] Add quality gate keys to `aura.config.json`

### Phase 2 — Goal Intelligence Router (P1)
- [ ] Build P1 workflow in n8n (classify + route + respond)
- [ ] Wire to existing AURA `/webhook/goal` for fast lane
- [ ] Wire to P3 stub for dual lane (P3 stub = placeholder webhook that returns `{status: 'pending'}` until Phase 3)

### Phase 3 — Pipeline Coordinator + Quality Gate (P3 + P2)
⚠️ P3 emits to P4/P5 — configure n8n HTTP nodes with "Continue on Error" until Phase 4 creates those webhooks.
- [ ] Build P3 (parallel fan-out, merge, score check, AURA trigger, poll)
- [ ] Build P2 (Dev Suite quality gate webhook, 3-agent review, Council Chair)
- [ ] Wire `before_act` hook in orchestrator.py

### Phase 4 — Feedback Loop + Observability (P4 + P5)
- [ ] Build P4 (lesson memory store, skill weight update, evolution trigger)
- [ ] Extend `_notify_n8n()` in reflector.py to POST to P4
- [ ] Build P5 (pipeline_runs collector, unified trace)

### Phase 5 — Hardening
- [ ] Add circuit breaker state to P3
- [ ] Add retry + backoff to all HTTP nodes in n8n
- [ ] Rotate `AGENT_API_TOKEN`, `MCP_API_TOKEN`, `N8N_MCP_TOKEN`
- [ ] Generate `N8N_WEBHOOK_SECRET` + add HMAC validation to all inbound webhooks

---

## Files Changed

| File | Change |
|------|--------|
| `core/orchestrator.py` | Add `_should_run_quality_gate()`, `_call_quality_gate()`, `before_act` hook |
| `aura_cli/server.py` | Add `POST /webhook/plan-review` endpoint |
| `agents/reflector.py` | Extend `_notify_n8n()` → POST to feedback_loop_webhook with skill_scores + pipeline_run_id |
| `aura.config.json` | Add quality_gate_enabled, quality_gate_webhook, feedback_loop_webhook, pipeline_coordinator_webhook, circuit_breaker_enabled |
| `n8n-workflows/workflow5_goal_router.js` | **NEW** P1 Goal Intelligence Router |
| `n8n-workflows/workflow6_pipeline_coordinator.js` | **NEW** P3 Pipeline Coordinator |
| `n8n-workflows/workflow7_quality_gate.js` | **NEW** P2 Dev Suite Quality Gate |
| `n8n-workflows/workflow8_feedback_loop.js` | **NEW** P4 Feedback Loop |
| `n8n-workflows/workflow9_observability.js` | **NEW** P5 Observability Collector |
| `n8n-workflows/workflow1_parallel_skills.js` | Update: retry logic, Eval Optimizer error fallback |
| `n8n-workflows/workflow4_goal_orchestrator.js` | Update: manualTrigger → webhookTrigger |
| `tests/test_pipeline_optimization.py` | **NEW** Integration tests for P1/P3/P4 webhooks |

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Dual-lane end-to-end (PR → plan reviewed → code applied → notified) | Works without human touch |
| Eval Optimizer gate accuracy | ≥ 90% of scores ≥ 0.75 lead to successful AURA cycles |
| Dev Suite quality gate latency | < 90s (parallel 3-agent review) |
| Fast lane latency overhead vs today | < 5% (classification only) |
| Pipeline run trace completeness | 100% of runs have pipeline_run_id in AURA + n8n |
| Circuit breaker | Opens correctly after 3 failures; closes after 5-min cooldown |
| Feedback loop lesson storage | Every AURA reflect posts to P4 within 2s |

---

## Open Questions

1. Should the `before_act` hook **block** the AURA cycle waiting for Dev Suite review, or run it asynchronously and inject critique on the **next** cycle? (Recommended: async with injection on same cycle if < 90s, else skip gate.)
2. Should P2 Dev Suite Quality Gate use all 7 agents or a lighter 3-agent subset for the plan-review use case? (Recommended: 3-agent subset — Code Architect + Security Auditor + Test Engineer.)
3. n8n's `Call Workflow Tool` node is available in Aura Dev Suite — should internal sub-workflow invocations use this instead of HTTP webhooks? (Recommended: yes for intra-n8n calls; HTTP only for AURA-facing calls.)
