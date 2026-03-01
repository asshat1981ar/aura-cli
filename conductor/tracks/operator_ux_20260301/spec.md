# Track Spec: Operator UX and Observability Hardening

## Goal
Make AURA's operator-facing surfaces feel like one coherent control plane. The CLI, TUI, metrics views, and structured logs should expose the same runtime truth with clear failure signals, stable contracts, and enough context to debug long-running autonomous sessions quickly.

## Scope
- **Runtime Surface Parity:** Align `goal status`, `metrics`, `watch`, Studio panels, and JSON-facing CLI output so queue state, cycle progress, and verification outcomes agree.
- **Structured Log Reliability:** Harden stream-first logging behavior across stdout/stderr consumers, long-running sessions, and operator tooling.
- **Operator-Focused Metrics:** Expand or normalize metrics/reporting so cycle outcomes, retries, verification failures, and queue health are easy to inspect.
- **TUI/Studio Usability:** Improve high-signal terminal UX for active runs, with emphasis on cycle visibility, queue state, and memory/metrics panels.

## Functional Requirements
- Runtime status commands and operator panels must expose a consistent set of queue, goal, and cycle fields.
- Structured logs must remain machine-parseable and operator-readable during interactive, watch, and long-running modes.
- Operator metrics must distinguish success, failure, skip, retry, and policy-stop outcomes clearly.
- TUI/Studio views must surface enough detail to triage failed goals without inspecting raw implementation internals by default.

## Non-Functional Requirements
- **Determinism:** Operator surfaces must derive from canonical runtime state, not duplicated ad hoc snapshots.
- **Performance:** Watch/TUI/metrics refreshes must remain responsive in terminal environments.
- **Contract Stability:** JSON outputs and snapshots must stay test-covered so CLI consumers do not drift silently.

## Acceptance Criteria
- A single autonomous run can be inspected consistently through CLI status, metrics output, and TUI/Studio panels.
- Log-stream handling is covered by tests for both stdout-capture and default runtime behavior.
- Snapshot and/or contract tests protect the operator-facing help and JSON surfaces that changed.
- Operator-facing commands surface actionable failure context for queue stalls, verification failures, and retry loops.

## Phase 1 Audit Summary
- `goal status` currently exposes queue/archive data plus capability bootstrap details, but it does not surface the active cycle, last verification result, or stop reason in JSON form.
- `metrics` currently derives from legacy `Brain.recall_recent()` strings shaped like `outcome:<id> -> <json>`, while the orchestrator now persists raw decision-log entries through `MemoryStore.append_log()`. That leaves metrics and live runs without a shared source of truth.
- `watch` and Studio panels derive queue state directly from `GoalQueue`, but cycle state is maintained in local TUI memory rather than a canonical runtime payload.
- The SSE goal stream already emits a compact cycle summary, but it synthesizes that shape independently and only attaches the final policy stop reason in the terminal `complete` event.

See `conductor/tracks/operator_ux_20260301/audit.md` for the full surface map and the proposed canonical queue/cycle contract.

## Out of Scope
- New MCP tools unrelated to runtime visibility.
- Large visual redesigns that discard existing TUI/Studio structure.
- Autonomous feature development unrelated to operator observability.
