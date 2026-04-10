# Changelog

All notable changes to AURA CLI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-10

### Added

#### Sprint 1: Server Decomposition
- `aura_cli/api/` — Modular FastAPI application structure
  - `app.py` — Thin composition root with lifespan management
  - `middleware/auth.py` — JWT authentication middleware
  - `routers/health.py` — Health, readiness, liveness endpoints
  - `routers/runs.py` — Pipeline execution and webhook endpoints
  - `routers/ws.py` — WebSocket endpoints for real-time updates

#### Sprint 3: JWT Hardening + Auth Router
- `aura_cli/api/routers/auth.py` — JWT auth endpoints: POST /api/v1/auth/login, /refresh, /logout
  - Login issues HS256 access (≤24h) and refresh tokens
  - Refresh exchanges refresh token for new access token
  - Logout revokes token JTI in SQLite blocklist
- `tests/test_jwt_hardening.py` — 22 integration tests (key length, HS256, expiry cap, JTI revocation)

#### Sprint 3: Sandbox Security Hardening
- Network blocking via proxy environment variables (`_SANDBOX_NETWORK_ENV`)
- Resource limits: 30s CPU, 512 MiB memory (`RLIMIT_CPU`, `RLIMIT_AS`)
- Filesystem restrictions via runtime `open()` wrapper
- Security audit document (`docs/security/sandbox-audit-v1.0.md`)

#### Sprint 4/5: Test Coverage
- `tests/agents/test_applicator_handler.py` — Unit tests for applicator handler
- `tests/agents/test_sandbox_unit.py` — Unit tests for sandbox module
- `tests/integration/test_e2e_sandbox_retry.py` — E2E tests for sandbox retry

#### Sprint 6: CLI Infrastructure
- `aura history [--limit N] [--json]` — List completed goals from GoalArchive (FR-011)
- `core/circuit_breaker.py` — Three-state circuit breaker for LLM resilience
- `core/cost_tracker.py` — Cost tracking with `AURA_COST_CAP_USD` enforcement
- `memory/redis_cache_adapter.py` — Optional Redis L0/L1 caching (activated via `REDIS_URL`)
- Prometheus metrics resilience for test reimports

#### Sprint 8: Infrastructure Hardening + Logging
- Docker Compose security: `no-new-privileges`, user restrictions, tmpfs
- Resource limits: mem_limit, cpus for all services
- Logging rotation: max-size 10m, max-file 3
- Pinned base images: nginx:1.25-alpine, redis:7.2-alpine

- Migrated 28+ `print()` calls to `logging` in agents and core library code

#### Sprint 9: Documentation + Coverage
- Production deployment guide (`docs/deployment/production-guide.md`) — 546 lines
- Kubernetes deployment manifests
- Security hardening checklist
- Backup & recovery procedures
- Coverage gate: `fail_under=8` scoped to `aura_cli`, `core`, `agents`, `memory`
- `tests/test_correlation.py` — 13 tests for CorrelationManager / TraceContext (98% coverage)
- `tests/test_config_schema.py` — 13 tests for ConfigValidator / validate_config (95% coverage)
- `tests/test_sanitizer.py` — Extended to 10 tests for sanitize_path / sanitize_command (95% coverage)

### Changed

- `agents/sandbox.py` — Wrapped code execution with filesystem restrictions
- `agents/skills/base.py` — Added `SKIP_DIRS` and `iter_py_files()` helpers
- `aura_cli/cli_options.py` — Improved CLI contract reporting
- `docker-compose.prod.yml` — Hardened production configuration
- `pyproject.toml` — Coverage scoped to 4 measured dirs; `fail_under=8`

### Fixed

- `tests/test_server_api.py` — TypeError in `test_metrics_has_skill_metrics` and `test_execute_run_persists_audit_entries` when `prometheus_client` is installed

### Security

- `pycryptodomex` upgraded 3.11.0 → 3.23.0 (fixes PYSEC-2024-3 Manger OAEP side-channel)
- JWT: algorithm:none confusion prevented; 256-bit minimum key; JTI revocation in SQLite
- Sandbox violations now logged to `aura_sandbox_violations_total` metric
- API token required for all mutating endpoints
- Cost cap prevents runaway LLM spending
- All subprocesses run with restricted network and filesystem access

### Compliance

- OWASP ASVS L1: Input Validation (V5.2) ✅
- OWASP ASVS L1: File Execution (V12.3) ✅
- NIST 800-53: Process Isolation (SC-39) ✅
- NIST 800-53: Identification and Authentication (IA-5) ✅ (JWT hardening)

## [Sprint S004] - 2026-03-27

### Added
- `core/swarm_models.py`: Pydantic dataclasses for hierarchical swarm workflow (SwarmTask, TaskResult, CycleLesson, CycleReport, PRGateDecision, SupervisorConfig, SDLCFinding)
- `agents/hierarchical_coordinator.py`: Architect→Coder→Tester async pipeline with lesson injection every N cycles
- `agents/sdlc_debugger.py`: 10-lens SDLC failure classifier (requirements, design, implementation, integration, testing, security, performance, operations, DX, delivery)
- `core/swarm_supervisor.py`: `install_swarm_runtime()` single-call wiring entry point
- `memory/learning_loop.py`: `LessonStore` JSONL persistence for cycle lessons
- `aura_cli/runtime_factory.py`: `RuntimeFactory` with `_WeaknessRemediatorLoop` (every 5 cycles) and `_ConvergenceEscapeLoop` (per cycle)
- Feature flag: `AURA_ENABLE_SWARM=1` activates hierarchical coordinator in `create_runtime()`
- 27 net-new tests across `test_swarm_supervisor.py`, `test_swarm_models.py`, `test_runtime_factory.py`

### Changed
- `aura_cli/cli_main.py`: `create_runtime()` now wires `install_swarm_runtime()` when `AURA_ENABLE_SWARM=1`
- `agents/hierarchical_coordinator.py`: `_perform_github_delivery` gated behind `SupervisorConfig.github_delivery_enabled` (default `False`)
- All bare `print()` calls replaced with `log_json()` in coordinator and debugger

### Safety
- `github_delivery_enabled: bool = False` in `SupervisorConfig` — no GitHub API calls without opt-in
- `AURA_SWARM_PR_GATE=1` required to enable PR automation

## [Unreleased] — Sprint S003: SADD Workflow Completion

### Added

- **`sadd resume --run`**: fully implements SADD session resume. Restores
  `WorkstreamGraph` from `SessionStore`, resets previously-failed and blocked
  workstreams to pending for retry, then drives the remaining workstreams to
  completion via `SessionCoordinator.resume()`. Without `--run`, prints a safe
  summary (total/completed/remaining) and exits 0.
- **`SessionCoordinator.resume(graph, completed_results)`**: new method that
  accepts a pre-restored graph and dict of completed results, then executes only
  the remaining workstreams. Resets `failed` and `blocked` nodes to `pending`
  before execution so dependency chains are correctly retried.
- **Live progress output** in `SessionCoordinator`: per-workstream start/complete/fail
  lines printed to stdout during live `sadd run` and `sadd resume` execution so
  users can see real-time progress.
- **`MCPToolBridge` auto-wiring**: `SessionCoordinator` now accepts an optional
  `mcp_bridge` parameter (passed through to `SubAgentRunner`). Both `sadd run`
  and `sadd resume` dispatch handlers auto-create `MCPToolBridge()` with
  `(ImportError, OSError)` fallback to `None`.
- **`tests/test_sadd_coordinator.py`**: 19 tests covering live coordinator
  execution, resume skipping completed workstreams, resume retrying failed
  workstreams, and unblocking dependents of failed workstreams.

### Fixed

- `sadd resume` dispatch handler was a stub ("not yet implemented"); it now
  executes the full resume path via `SessionCoordinator.resume()`.
- `_handle_sadd_resume_dispatch`: failed workstreams in raw results were
  incorrectly passed to `graph.mark_completed()`; fixed to call `mark_failed()`
  for failed results and only pass truly-completed results to `coordinator.resume()`.
- `SessionCoordinator.resume()`: blocked dependents of failed workstreams were
  not reset to pending, preventing them from executing after the failed dependency
  was retried; now resets both `failed` and `blocked` nodes before the execution loop.
- MCP bridge `except Exception` narrowed to `except (ImportError, OSError)` to
  avoid swallowing unexpected runtime errors.

### Added

- **`goal resume` command** (`core/in_flight_tracker.py`): recovers goals silently lost when the
  AURA loop is interrupted between queue dequeue and archive. Writes `memory/in_flight_goal.json`
  atomically after each dequeue; `goal resume` re-prepends the goal and optionally runs it.
  Closes #301.
- **`goal run --resume` flag**: automatically re-queues any interrupted in-flight goal before
  running the queue. `goal run` (no flag) prints a stderr warning if an in-flight file is detected.
- **`scripts/link_story_plans.py`**: validates that `plans_link` fields in ready-stage Forge stories
  point to existing plan files. Exits 1 on broken links. Integrates with the forge lint pipeline.
- **`scripts/new_story.py`**: scaffolds new Forge stories into `inbox/`. Supports `--quick` mode
  (8-field lightweight template) and full-template mode. Auto-increments story ID.
- **`.aura_forge/templates/story_quick.yaml`**: 8-field quick-ideation template for fast inbox
  capture without forcing early design commitment.
- **`scripts/lint_forge_index.py`**: backlog index drift detector (5 rule classes). Catches stories
  indexed but absent on disk, stories on disk but not indexed, phantom done entries, duplicate IDs
  across lanes, and broken plans_link references.

### Fixed

- `tests/test_orchestrator_hub.py`: added `pytest.importorskip("orchestrator_hub")` guard to
  prevent collection ERROR when the now-deleted `orchestrator_hub` package is absent.

### Changed

- `.aura_forge/backlog/ready/`: promoted AF-STORY-0006 (story-plan linker), AF-STORY-0008
  (follow-up status conventions), AF-STORY-0009 (quick ideation template) from refined to ready
  with full `contract_impact`, `safety_impact`, and `acceptance_criteria` declarations.
- `.aura_forge/schemas/story.schema.yaml`: added `design_pass_notes` typed section (6 pass types)
  and `plans_link` field.
- `docs/CLI_REFERENCE.md`: regenerated to include `goal resume` and `goal run --resume`.

---

## [0.1.0] — 2026-03-24

First stable release of AURA CLI — an autonomous, multi-agent software
development platform with persistent memory, adaptive planning, and
self-improvement capabilities.

### Highlights

- **13 innovation modules** spanning two development sprints, all wired into the orchestrator
- **187 tests** (118 unit + 69 integration) verifying end-to-end behaviour
- Full CI pipeline: Python 3.10 and 3.11 matrix, linting, snapshot contracts, and release automation

---

### Added

#### Sprint 1 — Core Innovation Modules

- **N-Best Code Generation** (`core/nbest.py`): generate *N* code candidates, score with a critic
  tournament, and promote the highest-quality implementation.
- **Phase Confidence Scoring** (`core/phase_result.py`): every orchestrator phase returns a
  `PhaseResult` with a 0–1 confidence score; `ConfidenceRouter` uses scores for data-driven
  routing instead of hard-coded pass/fail logic.
- **Experiment Tracking** (`core/experiment_tracker.py`): Karpathy-style measure-keep/discard
  discipline for tracking code-generation experiments and surfacing regressions.
- **A2A Protocol** (`core/a2a/`): Agent-to-Agent communication protocol with a server,
  client, task model, and `AgentCard` discovery — enabling multi-agent collaboration.
- **Lifecycle Hooks** (`core/hooks.py`): guaranteed-execution pre/post hooks for each
  orchestrator phase, with exit-code–based blocking semantics.
- **Memory Consolidation** (`core/memory_compaction.py`): automatic compaction loop that
  prevents `memory/decision_log.jsonl` from growing unboundedly while preserving recent history.
- **MCP Bi-directional Callbacks** (`core/mcp_events.py`): event-bus–style callback system
  for MCP servers to push updates back into the running orchestrator.

#### Sprint 2 — Planning and Analysis Modules

- **Tree-of-Thought Planning** (`core/tree_of_thought.py`): generate *N* plan candidates,
  score each with confidence, and select the best plan for the Act phase.
- **Code RAG** (`core/code_rag.py`): retrieve relevant past implementations from the Brain
  before code generation, augmenting the task bundle with working examples.
- **Skill Correlation Matrix** (`core/skill_correlation.py`): self-organising skill system
  that records co-activation patterns and suggests correlated skills for dispatch.
- **Team Coordinator** (`core/team_coordinator.py`): orchestrate multiple AURA agents on
  decomposed sub-goals and aggregate their results.
- **Quality Trend Analyzer** (`core/quality_trends.py`): detect quality regressions across
  cycles and automatically enqueue remediation goals.
- **AST Analyzer** (`agents/skills/ast_analyzer.py`): static analysis skill that extracts
  functions, classes, complexity metrics, and import graphs from Python source.

#### Testing

- Sprint 1 integration test suite (`tests/integration/test_innovation_integration.py`):
  30 tests covering HookEngine, ConfidenceRouter, NBestEngine, ExperimentTracker, A2AServer,
  EventBus, MemoryConsolidator, NegativeExampleStore, and cross-module flows.
- Sprint 2 integration test suite (`tests/integration/test_sprint2_integration.py`):
  20 tests covering TreeOfThought, CodeRAG, SkillCorrelation, QualityTrends, TeamCoordinator,
  and cross-module flows including orchestrator initialisation.

#### Infrastructure

- Release workflow (`.github/workflows/release.yml`): builds sdist + wheel, runs the full test
  suite against the built package, generates a changelog, creates the GitHub Release, and
  publishes to PyPI via trusted OIDC publishing.
- `pyproject.toml`: project metadata, `aura` console-script entry point, optional `dev`
  extras, and tool configuration for pytest, coverage, ruff, and bandit.

### Changed

- Orchestrator (`core/orchestrator.py`) now initialises and wires all 13 Sprint 1 and Sprint 2
  modules at startup, replacing the previous stub phase dispatching.

### Fixed

- `.gitignore` updated to exclude `.mcp.json` (MCP local config) and runtime memory artefacts.

---

[0.1.0]: https://github.com/asshat1981ar/aura-cli/releases/tag/v0.1.0
