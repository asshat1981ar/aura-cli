# Track Spec: BEADS-Orchestrator Convergence

## Goal

Add BEADS as AURA's structured decision layer while keeping `LoopOrchestrator` as the only execution authority. This track connects PRD context, conductor planning, and runtime execution through one canonical JSON bridge contract.

## Scope

- define BEADS bridge contracts
- add Node bridge adapter around `@beads/bd`
- build Python runtime wrapper and tests
- wire BEADS into orchestrator as a required gate in a later phase
- expose decision state to operator surfaces after gating is live

## Functional Requirements

- BEADS bridge must accept canonical JSON input and emit canonical JSON output.
- The orchestrator must be able to treat BEADS as a required gate.
- BEADS decisions must carry constraints, required tests, required skills, and follow-up goals.
- PRD and conductor context must be representable in the bridge payload.

## Non-Functional Requirements

- deterministic contract validation
- bounded bridge timeout
- no direct JS API coupling from Python callers
- rollout-safe scope control

## Acceptance Criteria

- a bridge contract and adapter exist with unit coverage
- BEADS PRD and track are the canonical planning sources
- the next implementation slice can wire the orchestrator without re-deciding the interface

## Out of Scope

- replacing existing orchestrator execution phases
- broad operator UI changes in this slice
- automatic migration of legacy PRDs
