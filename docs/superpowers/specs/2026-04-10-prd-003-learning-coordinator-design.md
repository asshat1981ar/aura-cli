# PRD-003: Autonomous Learning Loop — LearningCoordinator

**Date:** 2026-04-10
**Status:** Implemented — `feat/sprint-13-coverage`, ported from `claude/implement-ascm-v2-gSGye`
**Smoke test:** `artifacts=1`, `goals=[]` for low-severity cycle learning

---

## Problem Statement

AURA ran thousands of cycles but discarded the lessons from each one. Per-cycle learnings from `ReflectorAgent`, quality alerts from `DeepReflectionLoop`, and phase failure signatures all vanished between sessions, making every new run start from scratch.

## Goal

Wire per-cycle learnings into a persistent, structured artifact store. Surface actionable artifacts as new goals in the queue so AURA self-heals failure patterns without human intervention.

---

## Components

### 1. LearningArtifact (`core/learning_types.py`)

Canonical unit of what AURA learned from a single cycle.

| Field | Type | Description |
|-------|------|-------------|
| `artifact_id` | `str` | Auto-generated UUID hex |
| `cycle_id` | `str` | Source cycle identifier |
| `goal` / `goal_type` | `str` | What was being attempted |
| `artifact_type` | `str` | One of 5 categories (see below) |
| `insight` | `str` | Human-readable lesson |
| `evidence` | `dict` | Supporting raw data |
| `suggested_goal` | `str \| None` | Remediation goal text |
| `severity` | `str` | `low` / `medium` / `high` / `critical` |
| `acted_on` | `bool` | True once goal enqueued |

**`ARTIFACT_TYPES`** (frozenset):
- `phase_failure` — pipeline phase failing at high rate
- `skill_weakness` — skill produces low-signal output
- `quality_regression` — quality metric dropped below threshold
- `cycle_learning` — per-cycle learning string from ReflectorAgent
- `success_pattern` — pattern observed in successful cycles

**`SEVERITIES`** (tuple, ordered): `("low", "medium", "high", "critical")`

**Key methods:**
- `is_actionable() -> bool` — has `suggested_goal` and `not acted_on`
- `mark_acted_on()` — sets `acted_on = True`

**Tests:** 33 (`tests/test_learning_types.py`)

---

### 2. LearningCoordinator (`core/learning_coordinator.py`)

Aggregates learnings from all sources into `LearningArtifact` records and drains actionable ones into the goal queue.

**Public API:**

```python
coordinator = LearningCoordinator(memory_store, goal_queue)

# Called after every cycle completes
coordinator.on_cycle_complete(cycle_id, goal, goal_type, phase_outputs)

# Called during Phase 10 (discover) to drain actionable artifacts
count = coordinator.generate_backlog(limit=5)

# Query recent artifacts
artifacts = coordinator.get_recent_artifacts(limit=20)
```

**Internal flow:**
1. `on_cycle_complete()` — reads `ReflectorAgent` learnings + quality alerts from `phase_outputs`
2. `_sync_reflection_reports()` — polls `reflection_reports` memory tier via timestamp gating
3. Creates `LearningArtifact` for each insight above severity threshold
4. Persists to `learning_artifacts` memory tier
5. `generate_backlog()` — filters actionable artifacts, calls `goal_queue.batch_add()`, marks each as `acted_on`

**Integration points:**
- `core/orchestrator.py`: `attach_learning_coordinator()` method
- `core/orchestrator_learn.py`: `_record_cycle_outcome()` calls `on_cycle_complete()`; Phase 10 calls `generate_backlog()`
- `aura_cli/runtime_factory.py`: instantiated and attached in `_attach_advanced_loops()`

**Tests:** 34 (`tests/test_learning_coordinator.py`)

---

### 3. Supporting PRD-003 Types (`core/cycle_outcome.py`, `core/quality_snapshot.py`)

These modules existed pre-PRD-003 and are tested by `tests/test_learning_loop.py`:

| Module | Purpose | Tests |
|--------|---------|-------|
| `CycleOutcome` | Structured cycle result with JSON round-trip | 10 |
| `run_quality_snapshot()` | AST/test metrics for changed files | 10 |
| `AdaptivePipeline` | Win-rate tracking, parameter optimization | 8 |
| `AutonomousDiscovery` | Finding-to-goal translation, capped at 3 per drain | 11 |

**Total tests in `test_learning_loop.py`:** 39

---

## Test Coverage Summary

| File | Tests | Coverage Area |
|------|-------|---------------|
| `tests/test_learning_types.py` | 33 | `LearningArtifact`, `ARTIFACT_TYPES`, `SEVERITIES` |
| `tests/test_learning_coordinator.py` | 34 | `LearningCoordinator` all public methods |
| `tests/test_learning_loop.py` | 39 | `CycleOutcome`, `QualitySnapshot`, `AdaptivePipeline`, `AutonomousDiscovery` |
| **Total** | **106** | |

---

## Acceptance Criteria

- [x] `LearningArtifact.is_actionable()` — returns True only when goal set and not acted on
- [x] `LearningCoordinator.on_cycle_complete()` — produces at least 1 artifact for non-empty cycle insights
- [x] `LearningCoordinator.generate_backlog()` — drains actionable artifacts into goal queue and marks them acted_on
- [x] Smoke test: single low-severity cycle produces `artifacts=1`, `goals=[]` (below threshold, not enqueued)
- [x] All 106 tests pass

---

## Future Work

- Severity threshold configuration (currently `low` = always capture, but only `medium+` triggers backlog)
- Cross-cycle pattern detection: flag recurring failure signatures across N cycles
- Artifact TTL / pruning: prevent unbounded growth of `learning_artifacts` tier
- UI surface in `aura watch` / `aura logs` (PRD-005 TUI milestone)
