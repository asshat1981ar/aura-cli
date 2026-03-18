# RSI Dry-Run Audit Verification

**Date:** 2026-03-08
**Mode:** `dry_run`
**Runner:** `scripts/run_rsi_evolution.py`
**Report:** `/data/data/com.termux/files/usr/tmp/rsi_audit_50.json`
**Log:** `/data/data/com.termux/files/usr/tmp/rsi_audit_50.log`

## Summary

- Completed a 50-cycle RSI audit with `dry_run=true`.
- Processed all 50 cycles with no cycle-level exceptions.
- Provider readiness during the run: `chat_ready=true`, `embedding_ready=true`.
- The new dry-run guard in `EvolutionLoop.on_cycle_complete()` behaved as intended:
  - `evolution_runs=0`
  - `scheduled_triggers=0`
  - `hotspot_triggers=0`
  - log evidence: repeated `evolution_loop_skipped_dry_run`

## Report Snapshot

```json
{
  "status": "ok",
  "cycles_requested": 50,
  "dry_run": true,
  "processed_cycles": 50,
  "evolution_runs": 0,
  "failure_count": 0,
  "average_retry_count": 0.0,
  "proposal_count": 0
}
```

## Architectural Delta

- `complexity_scorer.high_risk_count`
  - baseline: `229.0`
  - current: `229.0`
  - delta: `0.0`

## Operational Findings

- The audit runner and summarizer are functioning end-to-end.
- The dry-run safety fix prevented unintended self-mutation during the long audit.
- Several analysis skills still show slow-path behavior:
  - skill timeouts were logged before later completion in some cycles
  - cycle durations remained high enough that a full live audit should be run in an isolated workspace, not the current dirty repo

## Post-Audit Timeout Triage

- `core.skill_dispatcher.dispatch_skills()` now:
  - caches read-only static-analysis skill results by project fingerprint
  - stops blocking on futures that already exceeded the dispatch timeout
  - allows late-finishing cacheable skills to warm the cache for subsequent cycles
- `agents/skills/architecture_validator.py` no longer performs a second full circular-import scan after `_analyse_project()`
- Focused validation after the fix:
  - `python3 -m pytest -q tests/test_skill_dispatcher.py tests/test_health_monitor.py tests/test_skills.py -k 'skill_dispatcher or health_monitor or architecture_validator or complexity_scorer or test_coverage_analyzer or refactoring_advisor or security_scanner'`
  - Result: `46 passed, 54 deselected`
- Real subset timing on the current repo for the heaviest refactor-analysis skills:
  - first dispatch: `36.26s`
  - second dispatch on unchanged state: `0.32s`

## Conclusion

This dry-run audit verifies the RSI runner, reporting path, provider readiness, and the dry-run safety contract.

It does **not** satisfy the track's final live-evolution requirement, because dry-run mode intentionally suppresses `EvolutionLoop` execution. The remaining open step is a non-dry-run 50-cycle audit in an isolated clone or worktree.
