# Track Spec: Stabilize Core and Finalize R4 Optimization

## Goal
The goal of this track is to harden the AURA core by finalizing the performance optimizations (R4 Recall) and ensuring architectural integrity through schema fixes and test stabilization.

## Scope
- **Metric Implementation:** Add the remaining Recall@k logic to the ASCM evaluation harness.
- **VectorStore Hardening:** Address schema edge cases in `core/vector_store.py` to prevent upsert failures.
- **WorkflowEngine Validation:** Ensure all new WorkflowEngine tests pass in the local environment and fix any discrepancies in state transitions.
- **Unified Doctor:** Finalize the transition to `aura_cli/doctor.py` by removing redundant root files.

## Success Criteria
- `tests/eval_ascm.py` reports valid Recall@k metrics.
- `VectorStore.upsert` handles all 15 columns without intermittent SQLite errors.
- 100% pass rate for `tests/test_workflow_engine_full.py`.
- `aura doctor` reports PASS for all system health checks.
