# Tech Debt Register

> Maintained by the AURA team.  Add new items as they are discovered; resolve by
> removing the row and referencing the commit/PR that fixed it.
>
> **Severity scale:** HIGH = blocks feature work or correctness; MEDIUM = slows
> development or harms maintainability; LOW = cosmetic / nice-to-have.

| ID | File | Description | Severity | Sprint |
|----|------|-------------|----------|--------|
| TD-001 | `aura_cli/api_server.py` | 2040 lines — monolithic API server.  Needs splitting into per-domain routers under `aura_cli/routers/`. | HIGH | Sprint 2 |
| TD-002 | `core/orchestrator.py` | 1990 lines — phase-runner logic embedded inline.  Extract each phase into a `core/phase_runners/` package to match the `PhaseDispatcher` abstraction already in place. | HIGH | Sprint 2 |
| TD-003 | `aura_cli/dispatch.py` | 1567 lines — all CLI command dispatch in one file.  Split by command group (goal, run, memory, mcp, …). | HIGH | Sprint 2 |
| TD-004 | `aura_cli/commands.py` | 1363 lines — same root cause as TD-003.  Should be split alongside `dispatch.py`. | HIGH | Sprint 2 |
| TD-005 | `core/model_adapter.py` | 1131 lines — multiple provider adapters (OpenAI, Anthropic, Ollama, …) co-located.  Extract each provider to `core/providers/<name>.py`. | HIGH | Sprint 3 |
| TD-006 | `aura_cli/cli_options.py` | 1034 lines — option group definitions bloated.  Split into per-command-group option files. | MEDIUM | Sprint 3 |
| TD-007 | `aura_cli/options.py` | 957 lines — overlaps with `cli_options.py`.  Consolidate and remove duplication. | MEDIUM | Sprint 3 |
| TD-008 | `core/workflow_engine.py` | 952 lines — step-runner logic should be extracted to `core/workflow_steps/`. | MEDIUM | Sprint 3 |
| TD-009 | `core/async_orchestrator.py` | 903 lines — large file with 8 pre-existing F401/F841 noqa suppressions.  Lints are suppressed, not fixed.  Extract phase handlers and remove noqa annotations. | MEDIUM | Sprint 3 |
| TD-010 | `core/context_graph.py` | 839 lines — graph algorithms (BFS, centrality, pruning) mixed with storage.  Extract to `core/graph_algorithms.py`. | MEDIUM | Sprint 4 |
| TD-011 | `tools/github_copilot_mcp.py` | 830 lines — MCP handler split needed.  Separate authentication, routing, and individual tool handlers. | MEDIUM | Sprint 4 |
| TD-012 | `core/agent_sdk/semantic_scanner.py` | 763 lines — scanner variants (AST, regex, embedding) should each live in their own module. | MEDIUM | Sprint 4 |
| TD-013 | `core/agent_sdk/tool_registry.py` | 752 lines — registry and loader logic co-located.  Split into `registry.py` + `loader.py`. | MEDIUM | Sprint 4 |
| TD-014 | `agents/brainstorming_bots.py` | 748 lines — each brainstorming bot persona is a candidate for its own module under `agents/brainstorming/`. | MEDIUM | Sprint 4 |
| TD-015 | `agents/multi_agent_workflow.py` | 724 lines — workflow steps should be extracted to match the handler-per-phase pattern used elsewhere. | MEDIUM | Sprint 4 |
| TD-016 | `agents/adversarial/strategies.py` | 652 lines — individual adversarial strategy classes should each have their own file. | MEDIUM | Sprint 5 |
| TD-017 | `core/capability_manager.py` | 641 lines — loader and resolver logic mixed.  Extract loaders into `core/capability_loaders/`. | MEDIUM | Sprint 5 |
| TD-018 | `agents/registry.py` | 633 lines — registry and lazy-loader logic in one file.  Split into `agents/registry.py` (spec declarations) and `agents/loader.py` (deferred import logic). | MEDIUM | Sprint 5 |
| TD-019 | `core/evolution_loop.py` | 750 lines — strategy runners embedded in the loop body.  Extract to `core/evolution_strategies/`. | MEDIUM | Sprint 5 |
| TD-020 | `core/voting/engine.py` | 583 lines — vote-counting and strategy dispatch should be separated. | MEDIUM | Sprint 5 |
| TD-021 | `agents/planner.py` | Line 116: `planner_output.dict()` uses the deprecated Pydantic v1 API.  Replace with `planner_output.model_dump()` (Pydantic v2). | HIGH | Sprint 2 |
| TD-022 | `core/hybrid_loop.py` | `HybridClosedLoop` is deprecated legacy code and emits deprecation warnings at construction.  Remove once all callers migrate to `LoopOrchestrator`. | MEDIUM | Sprint 2 |
| TD-023 | `core/test_goal_queue.py` | Tests instantiate the old SQLite-backed `GoalQueue` directly.  The queue has migrated to JSON-backed storage; tests need updating to reflect the current implementation. | HIGH | Sprint 2 |
| TD-024 | `aura_cli/server.py` | 802 lines — route modules should be extracted to match `api_server.py` refactor in TD-001. | MEDIUM | Sprint 3 |
| TD-025 | `tools/coverage_gap_analyzer.py` | 781 lines — analysis and report-generation logic co-located.  Extract reporters to `tools/coverage_reporters/`. | LOW | Sprint 5 |
| TD-026 | `tools/mcp_server.py` | 776 lines — follows the same MCP handler splitting pattern needed in TD-011. | LOW | Sprint 5 |
| TD-027 | `core/dpop.py` | 656 lines — DPoP token builders and verifiers should be split by token type. | LOW | Sprint 6 |
| TD-028 | `core/improvement_loop.py` | 525 lines — loop phases (propose, evaluate, apply) should each be extracted for testability. | LOW | Sprint 6 |
| TD-029 | `core/agentic_evaluation.py` | 521 lines — evaluator variants embedded inline.  Extract to `core/evaluators/`. | LOW | Sprint 6 |
| TD-030 | `tools/aura_control_mcp.py` | 510 lines — control handler split needed alongside other MCP server refactors. | LOW | Sprint 6 |
