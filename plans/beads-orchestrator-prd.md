# BEADS-Orchestrator Convergence PRD

## Summary

This PRD defines how AURA will adopt BEADS as both a planning framework and a required runtime gating layer while preserving `LoopOrchestrator` as the execution authority. The goal is to make high-value autonomous runs PRD-backed, decision-auditable, and safe to evolve without introducing a second competing executor.

## Problem

AURA already has strong execution machinery in `LoopOrchestrator`, but it lacks a mandatory structured decision layer between goal intake and execution. Product intent lives in PRDs and conductor tracks, while runtime decisions are made inside the orchestrator and agents. That creates a gap:

- goals can enter execution without a canonical product decision artifact
- conductor plans and runtime execution can drift apart
- operator audit trails explain what happened, but not always why the system allowed it

The repo already declares `@beads/bd`, but there is no bridge or product contract around it.

## Objectives

1. Establish BEADS as the canonical structured decision layer for gated runs.
2. Keep `LoopOrchestrator` as the sole execution engine for planning, coding, sandboxing, apply, verify, and reflection.
3. Connect PRD context, conductor track state, and runtime execution through one deterministic bridge contract.
4. Make BEADS decisions visible in logs, cycle summaries, and operator surfaces.

## Non-Goals

- replacing `LoopOrchestrator`
- introducing a JS-first runtime
- rewriting historical PRDs
- making BEADS optional for the initial gated scope

## User Journeys

1. A developer runs a high-risk goal. AURA assembles queue state, PRD context, and active track data, gets a BEADS decision, and only then enters planning.
2. BEADS blocks a run because required tests or constraints are missing. The operator sees a first-class blocked stop reason with a decision summary.
3. A future self-improvement loop can ask BEADS for revision guidance before another execution cycle begins.

## Functional Requirements

- A Python bridge must invoke `@beads/bd` through a stable JSON contract.
- Gated orchestrator runs must require a valid BEADS decision.
- BEADS decisions must support `allow`, `block`, and `revise`.
- AURA must persist normalized BEADS outputs into cycle history and operator views.
- BEADS integration must be configurable by scope and timeout.

## Runtime Model

- `LoopOrchestrator` remains the executor.
- BEADS runs before the plan phase for gated scopes.
- Invalid, unavailable, or blocking BEADS results halt the cycle deterministically.

## Delivery Phases

### Phase 1: Foundation
- PRD, conductor track, bridge contract, bridge adapter, unit tests

### Phase 2: Orchestrator Gate
- runtime wiring, stop reasons, cycle persistence, planner input

### Phase 3: Operator Surfaces
- CLI status, SSE, TUI/runtime snapshot visibility

### Phase 4: Rollout
- gated `goal_run`
- then broader runtime scopes after stabilization

## Acceptance Criteria

- BEADS bridge returns validated canonical JSON to Python callers
- gated runs cannot reach plan phase without valid BEADS allow decision
- blocked/revise/unavailable outcomes are observable in cycle summaries
- conductor and PRD context can be assembled into one bridge payload

## Risks

- BEADS package API instability
  - Mitigation: absorb all JS churn inside `scripts/beads_bridge.mjs`
- Runtime latency
  - Mitigation: timeout control and scoped rollout
- Drift between planning docs and runtime
  - Mitigation: PRD/track context included directly in bridge payload
