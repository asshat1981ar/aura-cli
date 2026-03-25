# AURA Self-Improvement via SADD

Bounded self-improvement spec for SADD to execute on AURA itself.

## Workstream: Code Quality Analysis

Analyze AURA core modules for code smells, complexity hotspots, and test gaps.

- Run complexity scoring on core/orchestrator.py, core/workflow_engine.py
- Identify functions with cyclomatic complexity > 10
- List modules with < 50% test coverage
- Acceptance: analysis report written to memory/sadd_analysis.json

## Workstream: Test Coverage Backfill

Add missing unit tests for under-tested SADD modules.

- Create tests/integration/test_sadd_e2e.py with dry-run e2e flow
- Add edge case tests for parser (malformed markdown, unicode, empty sections)
- Add concurrent execution tests for session coordinator
- Depends on: Code Quality Analysis
- Acceptance: test count increases, all new tests pass

## Workstream: Documentation Update

Update CLAUDE.md and CLI reference with SADD commands and architecture.

- Add SADD section to CLAUDE.md describing sadd-run, sadd-status, sadd-resume
- Regenerate CLI_REFERENCE.md
- Add core/sadd/ to the repository structure diagram
- Depends on: Test Coverage Backfill
- Acceptance: docs/CLI_REFERENCE.md is current, CLAUDE.md has SADD section
