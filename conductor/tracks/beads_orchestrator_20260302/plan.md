# Track Plan: BEADS-Orchestrator Convergence

## Phase 1: Planning and Bridge Foundation
- [ ] Task: Create the canonical BEADS-orchestrator PRD
  - [ ] Write `plans/beads-orchestrator-prd.md`
  - [ ] Capture authority boundaries between PRD, BEADS, and `LoopOrchestrator`
- [ ] Task: Create the bridge contract and adapter
  - [ ] Add `core/beads_contract.py`
  - [ ] Add `core/beads_bridge.py`
  - [ ] Add `scripts/beads_bridge.mjs`
  - [ ] Add bridge unit tests
- [ ] Task: Conductor - User Manual Verification 'Bridge Foundation' (Protocol in workflow.md)

## Phase 2: Orchestrator Gating
- [ ] Task: Add required BEADS gate to `LoopOrchestrator`
  - [ ] Run BEADS before the plan phase for gated scopes
  - [ ] Persist BEADS decisions into cycle history and stop reasons
- [ ] Task: Thread decision constraints into planner/synthesizer input
- [ ] Task: Conductor - User Manual Verification 'Orchestrator Gating' (Protocol in workflow.md)

## Phase 3: Runtime Surfaces and Rollout
- [ ] Task: Expose BEADS state in CLI, SSE, and operator snapshots
- [ ] Task: Add config and rollout controls for BEADS scopes/timeouts
- [ ] Task: Conductor - User Manual Verification 'Runtime Surfaces and Rollout' (Protocol in workflow.md)
