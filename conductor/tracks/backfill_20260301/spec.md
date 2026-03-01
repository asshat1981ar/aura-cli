# Track Spec: Autonomous Feature Backfill (Self-Prompt to Fill)

## Overview
This track introduces a goal-driven "Self-Prompt to Fill" mechanism. AURA will automatically detect if a high-level goal requires capabilities or coverage that are currently missing—specifically focusing on test coverage gaps—and will autonomously generate internal sub-goals to "backfill" those features before or alongside the primary task.

## Functional Requirements
- **Integrated Coverage Scanning:** Integrate with the `StructuralAnalyzerSkill` or a new specialized skill to identify source files with zero or low test coverage relevant to the current goal.
- **Dynamic Sub-Goal Generation:** When a goal is initiated, AURA must check relevant module health. If coverage is below the threshold, it must "self-prompt" to create a `test_backfill` sub-goal.
- **Coverage-Aware Planning:** The `PlannerAgent` must prioritize these backfill tasks if the system detects that implementing a new feature on untested code increases architectural risk.
- **Threshold Enforcement:** Define a configurable "Reliability Threshold" (e.g., 80% coverage) that triggers the self-prompt mechanism.

## Non-Functional Requirements
- **Performance:** Coverage scanning should be fast (< 2s) to avoid delaying the main orchestration loop.
- **Safety:** Backfill goals must adhere to the standard `Atomic Apply` safety policies.

## Acceptance Criteria
- Running a goal on a module with 0% coverage automatically generates a "Write missing tests" sub-task in the plan.
- The `StructuralAnalyzerSkill` correctly reports coverage gaps to the Orchestrator.
- The mechanism can be toggled via `aura.config.json` (`auto_backfill_coverage: true`).

## Out of Scope
- Automated implementation of complex business logic (focus is on infrastructure and tests).
- Modifying PRD/Roadmap files automatically.
