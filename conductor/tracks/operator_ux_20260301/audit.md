# Runtime Surface Audit

## Goal
Capture how operator-facing AURA surfaces currently represent queue and cycle state, then define one canonical payload shape that CLI status, metrics, watch/Studio, and streaming surfaces can all share.

## Current Surface Map

| Surface | Current source | Current shape | Drift / risk |
| --- | --- | --- | --- |
| `goal status --json` | `aura_cli.commands._handle_status()` | `queue_length`, `queue`, `completed_count`, `completed`, `capabilities` | Good queue snapshot, but no active cycle, last verification, or stop reason. |
| `queue list` | `GoalQueue.queue` | Text-only numbered list of pending goals | Separate queue formatting path from `goal status`; no reusable JSON contract. |
| `metrics` | `Brain.recall_recent()` filtered for `outcome:` strings | Legacy string parsing into `success`, `duration`, `goal`, `cycle_id` | Not backed by the current orchestrator persistence path; can drift or go blank even when runs exist. |
| `watch` / `studio` queue panel | `GoalQueue.queue` | Pending goal strings only | No completed work, active-cycle summary, or failure context. |
| `watch` / `studio` cycle panel | Local TUI callback state | `current_goal`, `current_phase`, per-phase icons, optional confidence | Not fed from a persisted canonical cycle object. |
| `watch` / `studio` metrics panel | Local `_cycle_log` in `AuraStudio` | `goal`, `success`, `duration_s`, `ts` | Orchestrator never emits `on_cycle_complete`, so this log is not a reliable source of truth. |
| SSE `goal` stream | `aura_cli.server.goal_stream()` | `start`, `health`, per-cycle `summary`, final `complete` event | Uses yet another hand-built summary shape and splits cycle vs final stop reasons across different events. |
| Decision log | `LoopOrchestrator._record_cycle_outcome()` | Raw `entry` with `cycle_id`, `goal_type`, `phase_outputs`, `stop_reason` | Missing `goal`, condensed verification summary, and operator-friendly queue/cycle fields. |

## Key Findings

1. Queue state is relatively consistent, but every surface formats it independently.
   `goal status`, `queue list`, and the queue panel all read the same queue, but there is no shared queue-summary object.

2. Cycle state has no canonical producer.
   The orchestrator returns raw cycle entries, SSE compresses them into a custom summary, and the TUI keeps a separate in-memory cycle log.

3. Metrics are still tied to a legacy storage contract.
   The metrics command expects `Brain` entries shaped like `outcome:<id> -> <json>`, while modern orchestration writes structured decision-log entries via `MemoryStore`.

4. The operator-facing cycle record is missing the user-visible goal string.
   `LoopOrchestrator._record_cycle_outcome()` persists `goal_type` and `phase_outputs`, but not the original `goal`.

5. The Studio autonomous worker is not aligned with the queue contract.
   It calls `goal_queue.pop()`, but `GoalQueue` exposes `next()`. Even aside from the method mismatch, the worker is bypassing a shared queue/cycle snapshot model.

## Canonical Contract Proposal

The operator surface should standardize on two reusable payloads:

1. `queue_summary`
2. `cycle_summary`

Everything else should either render these directly or compose them into a higher-level runtime snapshot.

### `queue_summary`

```json
{
  "pending_count": 2,
  "pending": [
    {
      "position": 1,
      "goal": "Fix failing snapshot tests"
    },
    {
      "position": 2,
      "goal": "Refresh CLI docs"
    }
  ],
  "completed_count": 1,
  "completed": [
    {
      "goal": "Stabilize runtime auth tests",
      "score": 8.5
    }
  ],
  "active_goal": "Fix failing snapshot tests",
  "updated_at": 1767340800.0
}
```

Required fields:
- `pending_count`
- `pending`
- `completed_count`
- `completed`
- `active_goal`
- `updated_at`

Notes:
- `pending[*].goal` stays string-based for now because `GoalQueue` stores strings today.
- `active_goal` may be `null` when idle.
- `completed[*].score` may be `null` if a producer cannot supply it.

### `cycle_summary`

```json
{
  "cycle_id": "cycle_abc123",
  "goal": "Fix failing snapshot tests",
  "goal_type": "bug_fix",
  "state": "running",
  "current_phase": "verify",
  "phase_status": {
    "ingest": "pass",
    "skill_dispatch": "pass",
    "plan": "pass",
    "critique": "pass",
    "synthesize": "pass",
    "act": "pass",
    "sandbox": "pass",
    "apply": "pass",
    "verify": "running",
    "reflect": "pending"
  },
  "verification_status": null,
  "stop_reason": null,
  "failures": [],
  "retry_count": 1,
  "applied_files": [
    "tests/snapshots/cli_help_top_level.txt"
  ],
  "failed_files": [],
  "queued_follow_up_goals": [],
  "started_at": 1767340800.0,
  "completed_at": null,
  "duration_s": null
}
```

Required fields:
- `cycle_id`
- `goal`
- `goal_type`
- `state`
- `current_phase`
- `phase_status`
- `verification_status`
- `stop_reason`
- `failures`
- `retry_count`
- `applied_files`
- `failed_files`
- `queued_follow_up_goals`
- `started_at`
- `completed_at`
- `duration_s`

Field rules:
- `state` is one of `idle`, `running`, or `complete`.
- `phase_status[*]` is one of `pending`, `running`, `pass`, `fail`, or `skip`.
- `verification_status` is one of `pass`, `fail`, `skip`, or `null` before verification completes.
- `stop_reason` is the canonical termination reason for the cycle or enclosing loop, not a surface-specific paraphrase.
- `queued_follow_up_goals` includes capability-backfill or other operator-visible follow-up goals created during the cycle.

### `operator_runtime_snapshot`

CLI status, watch/Studio, and SSE do not need separate bespoke schemas if they all compose the same top-level snapshot:

```json
{
  "schema_version": 1,
  "queue": "<queue_summary>",
  "active_cycle": "<cycle_summary or null>",
  "last_cycle": "<cycle_summary or null>"
}
```

This snapshot is the recommended source for:
- `goal status --json`
- watch/Studio panels
- SSE `cycle` and `complete` events
- future metrics aggregation

## Recommended Adoption Order

1. Teach `LoopOrchestrator` to emit and persist a canonical `cycle_summary`.
2. Build one queue-summary helper over `GoalQueue` + `GoalArchive`.
3. Make `goal status --json`, watch/Studio, and SSE consume the shared snapshot.
4. Repoint `metrics` at persisted `cycle_summary` records instead of legacy `Brain` string parsing.
5. Add contract tests for the shared snapshot and spot-check rendering surfaces against it.
