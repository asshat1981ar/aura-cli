# Track Plan: Stabilize Core and Finalize R4 Optimization

## Phase 1: Core Hardening & Metrics
- [x] Task: Finalize ASCM Evaluation Metrics
    - [x] Implement Recall@k (k=1, 3, 5) in `tests/eval_ascm.py`.
    - [x] Verify metrics against golden set vectors.
- [x] Task: VectorStore Schema & Upsert Fix
    - [x] Audit `core/vector_store.py` for any remaining column mismatches.
    - [x] Implement robust error handling for SQLite constraint violations.
- [x] Task: Conductor - User Manual Verification 'Core Hardening' (Protocol in workflow.md)

## Phase 2: Workflow & Infrastructure Stabilization
- [x] Task: WorkflowEngine Test Resolution
    - [x] Run `tests/test_workflow_engine_full.py` and debug any failures.
    - [x] Fix state transition logic in `core/workflow_engine.py` if required.
- [x] Task: Project Cleanup
    - [x] Remove redundant `test_aura_doctor_root.py`.
    - [x] Verify `aura doctor` integration.
- [x] Task: Conductor - User Manual Verification 'Infrastructure Stabilization' (Protocol in workflow.md)
