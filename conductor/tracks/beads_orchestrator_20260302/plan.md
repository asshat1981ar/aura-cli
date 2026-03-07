# Track Plan: BEADS-Orchestrator Convergence

## Phase 1: Planning and Bridge Foundation [checkpoint: 004b488]
- [x] Task: Create the canonical BEADS-orchestrator PRD
  - [x] Write `plans/beads-orchestrator-prd.md`
  - [x] Capture authority boundaries between PRD, BEADS, and `LoopOrchestrator`
- [x] Task: Create the bridge contract and adapter
  - [x] Add `core/beads_contract.py`
  - [x] Add `core/beads_bridge.py`
  - [x] Add `scripts/beads_bridge.mjs`
  - [x] Add bridge unit tests
- [x] Task: Conductor - User Manual Verification 'Bridge Foundation' (Protocol in workflow.md)

## Phase 2: Orchestrator Gating [checkpoint: 004b488]
- [x] Task: Add required BEADS gate to `LoopOrchestrator`
  - [x] Run BEADS before the plan phase for gated scopes
  - [x] Persist BEADS decisions into cycle history and stop reasons
- [x] Task: Thread decision constraints into planner/synthesizer input
- [x] Task: Conductor - User Manual Verification 'Orchestrator Gating' (Protocol in workflow.md)

## Phase 3: Runtime Surfaces and Rollout [checkpoint: 004b488]
- [x] Task: Expose BEADS state in CLI, SSE, and operator snapshots
- [x] Task: Add config and rollout controls for BEADS scopes/timeouts
- [x] Task: Conductor - User Manual Verification 'Runtime Surfaces and Rollout' (Protocol in workflow.md)

## Verification Notes
- Automated verification and the current manual verification script are recorded in `verification_20260307.md`.
- User approval received and checkpoint commit created: `004b488`.
