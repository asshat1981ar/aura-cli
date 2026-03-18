# Codebase Unfinished Work Plan

**Date:** 2026-03-18
**Status:** Completed
**Purpose:** Consolidate unfinished work in the repo into one grounded plan, then execute the highest-value stabilization slice immediately.

## Grounded Findings

1. The self-healing stack is only partially "done" in practice.
   Current code already contains `LintSkill`, `TestAndObserveSkill`, `HealthMonitor`, and skill-dispatch wiring, but one critical path is still incomplete:
   `HealthMonitor` currently dispatches with the `"default"` goal type, which skips the dedicated health skills it was meant to run.

2. The canonical test entrypoint is not stable.
   `pytest.ini` currently uses `testpaths = .`, which causes `python3 -m pytest` to collect:
   - vendored `llama.cpp` tests with missing optional dependencies
   - legacy malformed test artifacts such as `core/tests/test_aura.py`
   - production modules named like tests under `agents/skills/`

3. Several roadmap tracks remain open even though some underlying code exists.
   Since this plan was started, the BEADS track has been completed. The main
   remaining product-level tracks are:
   - `conductor/tracks/recursive_self_improvement_20260301/` final evolution verification
   - prototype consolidation around self-improvement/evolution modules

4. There is drift between planning docs and runtime reality.
   `plans/MASTER_DEVELOPMENT_PLAN.md` still describes major pieces as missing, but several of those pieces now exist and need verification, cleanup, and integration hardening rather than first-pass implementation.

## Execution Order

### Phase 1: Runtime Stabilization

- [x] Audit current code, plans, and tests
- [x] Identify concrete unfinished runtime gaps
- [x] Stabilize the canonical pytest contract
- [x] Fix `HealthMonitor` so it executes its health battery
- [x] Add regression coverage for both fixes

### Phase 2: Self-Healing Integration Hardening

- [x] Verify `HealthMonitor` integration with orchestrator-triggered loops
- [x] Reassess failure-routing behavior in `core/orchestrator.py`
- [x] Confirm `aura doctor`/health checks reflect current subsystem truth

Recent execution in this phase:
- `_attach_advanced_loops()` is now directly covered so `HealthMonitor` wiring keeps the orchestrator skill map, goal queue, memory store, and `project_root` intact.
- `SystemHealthProbe` now skips optional runtime checks that are not configured, so `aura doctor` no longer reports false subsystem warnings for missing `brain`, `memory_controller`, or `model_adapter` objects.
- `check_subsystem_probe()` centralizes doctor probe reporting and is covered by focused tests.
- `_route_failure()` now inspects `summary`, `stderr`, `error`, and `details` in addition to `failures`/`logs`, and recognizes more real external-failure signatures such as timeouts, connection failures, read-only filesystems, missing modules, and command lookup failures.

### Phase 3: Product Track Completion

- [x] Finish BEADS bridge/adapter/runtime gating work from `beads_orchestrator_20260302`
- [x] Build the RSI evolution verification harness for the open final track task
- [x] Consolidate or retire overlapping self-improvement prototype files per `plans/recursive-improvement-prototype-formalization.md`

Recent execution in this phase:
- Replaced the placeholder RSI verifier in `core/rsi_integration_verification.py` with a deterministic harness that can drive `EvolutionLoop.on_cycle_complete()` across recorded cycle history, count scheduled versus hotspot-triggered runs, measure proposal logging, and audit basic architectural metric drift.
- Added focused regression coverage for the new harness in `tests/test_rsi_integration_verification.py`.
- Completed a 50-cycle dry-run RSI audit with the new runner and recorded the result in `conductor/tracks/recursive_self_improvement_20260301/verification_20260308.md`.
- Triaged the dominant live-audit bottleneck by making `dispatch_skills()` cache unchanged static-analysis results by project fingerprint and stop blocking on already-timed-out futures; the heavy refactor-analysis subset improved from `36.26s` on first dispatch to `0.32s` on the second dispatch in the same process.
- Updated `conductor/tracks/recursive_self_improvement_20260301/plan.md` so the harness and dry-run audit are marked complete while the live 50-cycle autonomous audit remains open.

### Phase 4: Cleanup and Documentation Alignment

- [x] Decide whether legacy artifacts like `core/tests/test_aura.py` should be repaired, migrated, or retired (Archived to `archive/`)
- [x] Refresh high-level planning docs so "missing" vs "implemented but incomplete" is accurate (Updated `MASTER_DEVELOPMENT_PLAN.md`)
- [x] Prune or quarantine untracked prototype/runtime scratch files that are not part of the canonical product path (Cleaned up `.rsi_pid`, `coverage.json`, and dead scripts)

## Acceptance Criteria For This Execution Slice

1. `python3 -m pytest` no longer fails during collection because of vendored or malformed non-canonical tests.
2. `HealthMonitor.run_scan()` dispatches the provided health skills and can generate remediation goals from real dispatched results.
3. Targeted regression tests for health monitoring and skill dispatch pass locally.

## Recent Verification

- `python3 -m pytest -q tests/test_health_monitor.py tests/test_improvement_loops.py tests/test_aura_doctor.py`
  - Result: `33 passed`
- `python3 -m pytest -q tests/test_cli_main_dispatch.py -k attach_advanced_loops`
  - Result: `2 passed, 59 deselected`
- `python3 -m pytest -q tests/test_orchestrator_failure_routing.py tests/test_health_monitor.py tests/test_improvement_loops.py tests/test_aura_doctor.py`
  - Result: `46 passed, 15 subtests passed`
- `python3 -m pytest -q tests/test_rsi_integration_verification.py tests/test_evolution_loop_rsi.py tests/test_recursive_improvement.py`
  - Result: `13 passed`
- `python3 -m pytest -q tests/test_skill_dispatcher.py tests/test_health_monitor.py tests/test_skills.py -k 'skill_dispatcher or health_monitor or architecture_validator or complexity_scorer or test_coverage_analyzer or refactoring_advisor or security_scanner'`
  - Result: `46 passed, 54 deselected`
- Live smoke:
  - `SystemHealthProbe().run_all()` now returns `all_ok=True` with the default runtime-agnostic check set.
  - `run_doctor_v2(rich_output=False)` now reports `Subsystem probe: PASS`.
  - `scripts/run_rsi_evolution.py --cycles 50 --dry-run` completed with `processed_cycles=50`, `cycle_errors=0`, and `evolution_runs=0` as expected under the new dry-run guard.
  - real dispatcher timing on unchanged repo state for `symbol_indexer`, `complexity_scorer`, `code_clone_detector`, `tech_debt_quantifier`, and `refactoring_advisor`: `36.26s` first run, `0.32s` second run with cache hits.
