# Track Plan: Operator UX and Observability Hardening

## Phase 1: Runtime Surface Parity
- [x] Task: Audit Operator Surface Contracts
    - [x] Compare `goal status`, `metrics`, `watch`, and Studio/TUI panels for queue, cycle, and verification field parity.
    - [x] Identify duplicated or conflicting runtime state derivations in CLI/TUI code paths.
    - [x] Document findings in `conductor/tracks/operator_ux_20260301/audit.md`.
- [x] Task: Normalize Queue and Cycle Status Payloads
    - [x] Define a canonical runtime payload shape for operator-facing queue/cycle summaries.
    - [x] Update CLI/TUI entrypoints to consume the canonical shape instead of local one-off formatting where practical.
- [x] Task: Conductor - User Manual Verification 'Runtime Surface Parity' (Protocol in workflow.md)

## Phase 2: Log and Metrics Reliability
- [x] Task: Harden Structured Log Streaming
    - [x] Add or refine tests for `AURA_LOG_STREAM`, interactive runs, and stream consumers that parse JSON lines.
    - [x] Verify operator workflows capture actionable failure events without depending on incidental stderr behavior.
- [x] Task: Clarify Metrics and Stop Reasons
    - [x] Ensure operator metrics distinguish pass, fail, skip, retry, and policy-stop outcomes cleanly.
    - [x] Add tests for cycle outcome aggregation and metrics rendering where current coverage is thin.
- [x] Task: Conductor - User Manual Verification 'Log and Metrics Reliability' (Protocol in workflow.md)

## Phase 3: TUI and Studio Usability
- [x] Task: Improve High-Signal Operator Panels
    - [x] Review queue, cycle, memory, and metrics panels for missing failure context or noisy duplication.
    - [x] Prioritize visibility for current goal, last verification result, stop reason, and queued follow-up work.
- [x] Task: Refresh Operator Docs and Snapshots
    - [x] Update generated docs and snapshots for any intentional CLI/TUI/operator-contract changes.
    - [x] Verify operator entrypoints remain documented in `README.md` / generated CLI reference as needed.
- [x] Task: Conductor - User Manual Verification 'TUI and Studio Usability' (Protocol in workflow.md)
