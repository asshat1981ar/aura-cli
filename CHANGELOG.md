# Changelog

All notable changes to AURA CLI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
