# AURA v0.1.0 — Complete Release Notes

## 🚀 Overview

AURA v0.1.0 represents the first production-ready release of the Autonomous Unified Reasoning Agent CLI framework. This release includes all innovation modules from Sprint 1 and Sprint 2, comprehensive testing infrastructure, and multi-environment MCP architecture.

## ✨ What's New in v0.1.0

### Sprint 1 Innovation Modules (7 modules)

1. **N-Best Code Generation** - Generate multiple candidate solutions with temperature diversity
2. **Phase Confidence Scoring** - Adaptive phase execution based on confidence thresholds
3. **Experiment Tracking** - Persistent JSONL-based experiment recording with rotation
4. **A2A Protocol** - Agent-to-Agent communication protocol on port 8010
5. **Lifecycle Hooks** - Pre/post phase hooks for custom orchestration logic
6. **Memory Consolidation** - Automatic memory cleanup with retention thresholds
7. **MCP Bi-directional Callbacks** - Event-driven callbacks from MCP servers

### Sprint 2 Innovation Modules (6 modules)

1. **Tree-of-Thought Planning** - Multi-candidate plan generation with confidence-based critique skipping
2. **Code RAG** - Retrieval-augmented generation using Brain semantic memory
3. **Skill Correlation Matrix** - Discover and leverage skill correlations for better dispatch
4. **Team Coordinator** - Decompose goals into parallel sub-goals
5. **Quality Trend Analyzer** - Track cycle quality and auto-generate remediation goals
6. **AST Analyzer** - Python AST parsing and code structure analysis

### Multi-Environment MCP Architecture (PR #239)

- **MCP Hub Orchestrator** - Central coordination for 7 specialized MCP servers
- **Environment-Specific Servers**:
  - Gemini CLI (ports 8020-8029)
  - Claude Code (ports 8030-8039)
  - Codex CLI (ports 8040-8049)
- **Investigation & Remediation Agents** - Autonomous error analysis and fix generation
- **50+ New Tests** - Comprehensive coverage for MCP infrastructure

## 🧪 Testing Infrastructure

### Unit Tests
- **118 tests** covering all 13 innovation modules
- Full coverage of N-Best, confidence scoring, ToT, RAG, correlation matrix, and more

### Integration Tests
- **69 total integration tests** (30 Sprint 1 + 20 Sprint 2 + 19 existing)
  - Cross-module workflows (ToT → confidence → quality trends)
  - RAG + N-Best combined pipelines
  - Skill correlation with regression detection
  - Orchestrator end-to-end flows
- **1 skipped test** (pre-existing environment dependency)

### CI/CD
- Python 3.10 and 3.11 test matrix
- Lint, format, and security scanning (advisory mode)
- GitHub Actions workflows for Copilot, Codex, and Gemini automation
- Claude Code Review integration

## 🏗️ Core Framework

### Agent Architecture
- **16 specialized agents** covering orchestration, architecture, coding, QA, DevOps, security, documentation, dependency management, performance, code review, accessibility, onboarding, incident response, cost optimization, database migration, and API integration
- Agent configuration using `.agent.md` frontmatter specification
- Adaptive routing with EMA-based model selection

### Orchestration
- **10-phase autonomous loop**: ingest → skill dispatch → plan → critique → synthesize → act → sandbox → apply → verify → reflect
- Schema validation with lenient/strict modes
- Convergence detection and escape logic
- Human-in-the-loop approval gates

### Memory & State
- SQLite-backed semantic memory with NetworkX graph (`memory/brain.py`)
- JSONL decision log with rotation (`memory/store.py`)
- JSON-backed goal queue with priority ordering
- Skill weight adaptation based on cycle outcomes

### Model Routing
- Multi-provider support: OpenRouter, OpenAI, Gemini CLI, local models
- Task-specific routing (code_generation, planning, analysis, critique, embedding)
- Android local model profiles for edge deployment

## 📋 Project Infrastructure

### Documentation
- MIT License
- Contributing guidelines (CONTRIBUTING.md)
- Code of Conduct (Contributor Covenant 2.1)
- Security policy with vulnerability reporting (SECURITY.md)
- Comprehensive README with quick start and architecture overview
- CLI reference (auto-generated from code)
- Integration map (INTEGRATION_MAP.md)

### Issue & PR Templates
- Bug report template with reproduction steps
- Feature request template with problem/solution structure
- Pull request template with test plan requirements
- 28 categorized labels (priority, type, status, agent)

### Branch Protection
- CODEOWNERS for automatic reviewer assignment
- Required reviews and status checks
- Branch ruleset enforcement

## 📦 Installation

```bash
pip install aura-cli
```

### Development Setup

```bash
git clone https://github.com/asshat1981ar/aura-cli.git
cd aura-cli
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

## 🔧 Configuration

Configuration via `aura.config.json` and environment variables:

- `AURA_API_KEY` - OpenRouter API key
- `AURA_DRY_RUN=1` - No file or memory writes
- `AURA_SKIP_CHDIR=1` - Keep current working directory
- `AURA_STRICT_SCHEMA=1` - Abort cycle on schema validation failure
- `AURA_MODEL_ROUTING_<SUBKEY>` - Override individual model routing entries

## 🚀 Quick Start

```bash
# Add a goal to the queue
python3 main.py goal add "Refactor goal queue" --run

# Run a one-off goal (bypasses queue)
python3 main.py goal once "Summarize repo" --dry-run

# Check system health
python3 main.py doctor

# Start HTTP API server (FastAPI on port 8001)
uvicorn aura_cli.server:app --port 8001
```

## 📊 Test Coverage Summary

| Component | Unit Tests | Integration Tests |
|-----------|------------|-------------------|
| Sprint 1 Modules | 70 | 30 |
| Sprint 2 Modules | 48 | 20 |
| Core Orchestrator | - | 12 |
| MCP Infrastructure | 50+ | 7 |
| **Total** | **118+** | **69** |

## 🔗 Related Pull Requests

- #206 - Sprint 1 innovation modules + 30 integration tests
- #213 - Sprint 2 innovation modules (ToT, RAG, correlation, trends, AST)
- #214 - Wire Sprint 2 modules into orchestrator
- #219 - Sprint 2 integration tests (20 tests, closed as completed)
- #239 - Multi-environment MCP architecture

## 🗺️ Roadmap

See our [milestones](https://github.com/asshat1981ar/aura-cli/milestones) for upcoming releases:

- **v0.2.0** (Q2 2026) — JSON-RPC/stdio transport, RAG self-improvement, plugin system
- **v1.0** (June 2026) — Foundation Release with stable APIs
- **v1.1** (September 2026) — Agent Intelligence & Observability

## 🐛 Known Issues

- 2 pre-existing environment test failures (pip audit, python binary) - advisory only
- Python 3.12+ compatibility pending (currently tested on 3.10, 3.11)

## 🙏 Acknowledgments

This release represents the culmination of collaborative work across multiple AI coding agents (Copilot, Codex, Gemini, Claude) orchestrated through the AURA framework itself.

---

> This is a pre-release. APIs and agent interfaces may change before v1.0.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
