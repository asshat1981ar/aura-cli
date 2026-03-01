# Track Plan: Stabilize Core and Finalize R4 Optimization

## Phase 1: Core Hardening & Metrics
- [ ] Task: Finalize ASCM Evaluation Metrics
    - [ ] Implement Recall@k (k=1, 3, 5) in `tests/eval_ascm.py`.
    - [ ] Verify metrics against golden set vectors.
- [ ] Task: VectorStore Schema & Upsert Fix
    - [ ] Audit `core/vector_store.py` for any remaining column mismatches.
    - [ ] Implement robust error handling for SQLite constraint violations.
- [ ] Task: Conductor - User Manual Verification 'Core Hardening' (Protocol in workflow.md)

## Phase 2: Workflow & Infrastructure Stabilization
- [ ] Task: WorkflowEngine Test Resolution
    - [ ] Run `tests/test_workflow_engine_full.py` and debug any failures.
    - [ ] Fix state transition logic in `core/workflow_engine.py` if required.
- [ ] Task: Project Cleanup
    - [ ] Remove redundant `test_aura_doctor_root.py`.
    - [ ] Verify `aura doctor` integration.
- [ ] Task: Conductor - User Manual Verification 'Infrastructure Stabilization' (Protocol in workflow.md)
