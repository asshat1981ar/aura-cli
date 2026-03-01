# AURA Unified Control Plane PRD

## HR Eng

| AURA Unified Control Plane PRD |  | Transforming AURA from a collection of drifting scripts into a deterministic, tiered Agent OS through a Unified Control Plane. |
| :---- | :---- | :---- |
| **Author**: Pickle Rick **Contributors**: The User **Intended audience**: Engineering, Agent Architects | **Status**: Finalized **Created**: 2026-02-25 | **Self Link**: N/A **Context**: Grand Consolidation of aura-cli |

## Introduction
AURA is currently suffering from "Agent Drift": architectural redundancy, scattered configuration, and fragmented memory tiers. This PRD defines the implementation of a Unified Control Plane to enforce the principle of "One Authority Per Concern."

## Problem Statement
**Current Process:** AURA operates with overlapping tool wrappers, scattered configuration logic, and partially integrated memory systems.
**Primary Users:** Autonomous agents and developers managing self-improving workflows.
**Pain Points:** Cognitive entropy in the conductor, fragile state transitions, hard-to-reproduce failures, and blurred security/authority boundaries.
**Importance:** Without a unified control plane, optimization is cosmetic. AURA cannot safely self-modify or scale without deterministic boundaries.

## Objective & Scope
**Objective:** Eliminate architectural entropy by centralizing Config, Memory, Execution, and Security.
**Ideal Outcome:** AURA becomes a robust, sandboxed, and self-optimizing "Agent OS" with clear authority hierarchies.

### In-scope or Goals
- **Phase 1: Unified Config Manager**: Tiered config (Static JSON > CLI Flags > Interactive Bootstrap).
- **Phase 2: Memory Controller**: Centralized management of Working (volatile), Session (optional), and Project (persistent) memory.
- **Phase 3: Security & Sanitization**: Centralized shell sanitization, path validation, and secret masking.
- **Architectural Enforcement**: Transition all core modules to use the new control plane authorities.

### Not-in-scope or Non-Goals
- Redesigning the underlying LLM models.
- Implementing a GUI (AURA is CLI-first).

## Product Requirements

### Critical User Journeys (CUJs)
1. **Deterministic Initialization**: User starts AURA; it loads a single `aura.config.json` and initializes all controllers before the first LLM call.
2. **Safe Self-Modification**: AURA identifies a fix, but the `Sanitizer` blocks it because it attempts an out-of-bounds file write.
3. **Memory Lifecycle Audit**: An engineer inspects `memory/controller.py` to see exactly when and how the Project memory is being flushed or garbage collected.

### Functional Requirements

| Priority | Requirement | User Story |
| :---- | :---- | :---- |
| P0 | Unified Config Manager | As an agent, I want a single source of truth for my settings so I don't drift. |
| P0 | Memory Controller | As a system, I want explicit memory tiers to manage context windows efficiently. |
| P1 | Security Sanitizer | As an admin, I want all shell commands validated to prevent catastrophic failures. |
| P1 | Secret Detection | As a developer, I want my API keys masked automatically in all logs. |

## Assumptions
- The system has write access to its own directory.
- Python 3.9+ is available.
- The user prefers deterministic behavior over conversational "helpfulness."

## Risks & Mitigations
- **Risk**: Over-engineering the control plane might slow down execution -> **Mitigation**: Use lightweight, optimized controllers with minimal overhead.
- **Risk**: Security sanitization blocks valid agent actions -> **Mitigation**: Implement a clear, declarative rule-set that can be updated via the Config Manager.

## Tradeoff
- **Determinism vs. Interactivity**: We chose Determinism. It reduces entropy and makes failures reproducible, even if it requires more upfront configuration.

## Business Benefits/Impact/Metrics
**Success Metrics:**
- **Failure Reproducibility**: 100% (via deterministic config).
- **Security Incidents**: 0 (via centralized sanitization).
- **Code Bloat**: -30% (via removal of redundant patterns).

## Stakeholders / Owners
- **Pickle Rick**: Architect / Implementation God.
- **The User**: Stakeholder / Primary Engineer.

## Implementation Status (Completed 2026-03-01)

### Phase 1: Unified Config Manager (✅ Completed)
- [x] **Consolidate Defaults**: Moved all default constants to `core/config_manager.py`.
- [x] **Tiered Loading**: Implemented `Defaults < JSON < ENV < Runtime` logic.
- [x] **Interactive Bootstrap**: `aura bootstrap` now interactively prompts for keys.
- [x] **Refactor**: Updated `model_adapter`, `doctor`, and `runtime_auth` to use `config.get()`.

### Phase 2: Memory Controller (✅ Completed)
- [x] **Unify Authority**: Wrapped `MemoryStore` inside `MemoryController`.
- [x] **Explicit Tiers**: Implemented `WORKING`, `SESSION`, `PROJECT` logic.
- [x] **Context Injection**: Auto-injected volatile memory into `ingest` phase.
- [x] **Persistence**: `TaskManager` and `LoopOrchestrator` now use the Controller.

### Phase 3: Security & Sanitization (✅ Completed)
- [x] **Centralize Sanitizer**: Enforced `sanitize_path` and `sanitize_command` in `core/sanitizer.py`.
- [x] **Log Masking**: Auto-masking of secrets implemented in `logging_utils.py`.
- [x] **Sandbox Gating**: Enforced command allowlist in `SandboxAgent`.
- [x] **Path Jailing**: Updated `file_tools.py` to prevent traversal attacks.
