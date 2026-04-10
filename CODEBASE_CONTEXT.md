# AURA CLI - Codebase Context

> **Autonomous software development platform with multi-agent loop**
> 
> Generated: 2026-04-10 | Version: 0.1.0 | Repository: asshat1981ar/aura-cli

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [Key Modules](#key-modules)
5. [Agent System](#agent-system)
6. [Configuration](#configuration)
7. [Development Workflow](#development-workflow)
8. [Testing](#testing)
9. [Dependencies](#dependencies)
10. [Environment Variables](#environment-variables)

---

## Project Overview

AURA CLI is an autonomous software development platform that processes natural-language goals into code changes through a **10-phase pipeline**:

```
ingest → plan → critique → synthesize → act → sandbox → verify → reflect → apply → [repeat]
```

Each phase is owned by a dedicated agent - a Python class that accepts a dict, does work, and returns a dict. The `LoopOrchestrator` drives the loop, and the `TypedAgentRegistry` resolves the right agent for each phase at runtime.

### Key Features

- **Multi-Agent Orchestration**: 45+ specialized agents for different tasks
- **MCP Integration**: Model Context Protocol servers for tool access
- **SADD (Sub-Agent Driven Development)**: Parallel workstream execution
- **Semantic Memory**: Vector-based memory with embeddings
- **Telemetry**: Structured JSON logging with distributed tracing
- **Web UI**: React-based dashboard for monitoring
- **n8n Integration**: Workflow automation hooks

---

## Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Layer                                │
│  (main.py → aura_cli/cli_main.py → aura_cli/dispatch.py)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Core Orchestration                            │
│                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │   GoalQueue  │◄──►│LoopOrchestrator│◄──►│  MemoryStore │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│                              │                                   │
│                    10-Phase Pipeline                             │
│   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      │
│   │ingest│►│plan │►│critique│►│synth│►│act │►│sandbox│►│apply│►│verify│►│reflect│
│   └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Agents                                    │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐        │
│  │  Planner  │ │  Critic   │ │   Coder   │ │  Verifier │        │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### Core Patterns

| Pattern | Implementation |
|---------|---------------|
| **Agent Pattern** | `agents/base.py` - ABC with `run(input_data: Dict) -> Dict` |
| **Adapter Pattern** | Registry wraps agents for pipeline compatibility |
| **Capability-Based Resolution** | Agents resolved by capability tags |
| **Lazy Loading** | Agents loaded via `_AGENT_MODULE_MAP` |
| **Structured Outputs** | Pydantic schemas with Chain-of-Thought |

---

## Directory Structure

```
/home/westonaaron675/aura-cli/
├── main.py                    # CLI entry shim
├── run_aura.sh                # Bash convenience wrapper
├── pyproject.toml             # Python package configuration
├── aura.config.json           # Runtime configuration
├── settings.json              # Model routing and loop settings
├── .mcp.json                  # MCP server configuration
│
├── aura_cli/                  # CLI layer package (~30 modules)
│   ├── cli_main.py            # Command dispatcher
│   ├── dispatch.py            # Command dispatch registry
│   ├── commands.py            # Command handlers
│   ├── doctor.py              # System diagnostics
│   ├── server.py              # FastAPI server
│   ├── api/                   # API routers and middleware
│   └── tui/                   # Terminal UI components
│
├── core/                      # Orchestration engine (~140 modules)
│   ├── orchestrator.py        # Main 10-phase loop
│   ├── async_orchestrator.py  # Async pipeline implementation
│   ├── model_adapter.py       # LLM interface
│   ├── file_tools.py          # Atomic file operations
│   ├── goal_queue.py          # Persistent goal queue
│   ├── sadd/                  # Sub-Agent Driven Development
│   │   ├── types.py
│   │   ├── workstream_graph.py
│   │   ├── sub_agent_runner.py
│   │   └── session_coordinator.py
│   ├── agent_sdk/             # Agent SDK meta-controller
│   │   ├── controller.py
│   │   ├── tool_registry.py
│   │   └── resilience.py
│   └── metrics/               # Telemetry and analytics
│
├── agents/                    # Specialized agents (~45 agents)
│   ├── base.py                # Agent ABC base class
│   ├── registry.py            # Agent factory
│   ├── planner.py             # Plan generation
│   ├── critic.py              # Adversarial review
│   ├── coder.py               # Code generation
│   ├── verifier.py            # Test/lint verification
│   ├── reflector.py           # Outcome reflection
│   ├── innovation_swarm.py    # Multi-agent brainstorming
│   ├── skills/                # 40+ static analysis skills
│   │   ├── ast_analyzer.py
│   │   ├── complexity_scorer.py
│   │   ├── test_coverage_analyzer.py
│   │   └── security_scanner.py
│   └── handlers/              # Handler dispatch layer
│
├── memory/                    # State persistence
│   ├── brain.py               # SQLite semantic memory
│   ├── store.py               # JSONL decision log
│   ├── vector_store.py        # Semantic embeddings
│   └── embedding_provider.py  # Local embedding generation
│
├── tools/                     # MCP servers
│   ├── mcp_server.py          # Main MCP server (port 8001)
│   ├── sadd_mcp_server.py     # SADD operations (port 8020)
│   └── aura_control_mcp.py    # Control plane (port 8003)
│
├── tests/                     # Test suite (400+ test files)
│   ├── snapshots/             # JSON snapshot contracts
│   ├── integration/           # Integration tests
│   └── conftest.py            # pytest configuration
│
├── docs/                      # Architecture documentation
├── web-ui/                    # React-based dashboard
├── n8n-workflows/             # n8n workflow definitions
└── infra/                     # Infrastructure configs
```

---

## Key Modules

### Entry Points

| File | Purpose |
|------|---------|
| `main.py` | CLI shim - lazy loader delegating to `aura_cli.cli_main:main()` |
| `run_aura.sh` | Bash wrapper with aliases (`run`, `once`, `add`, `status`) |
| `aura_cli/cli_main.py` | Main command dispatcher + runtime initialization |

### Core Orchestration

| Module | Purpose |
|--------|---------|
| `core/orchestrator.py` | **LoopOrchestrator** - Main 10-phase autonomous loop |
| `core/async_orchestrator.py` | Async pipeline with non-blocking execution |
| `core/model_adapter.py` | LLM interface (Gemini, OpenAI, OpenRouter, local) |
| `core/mcp_agent_registry.py` | Typed registry for capability-based agent resolution |
| `core/file_tools.py` | Atomic file operations with overwrite safety |
| `core/goal_queue.py` | Persistent JSON-backed goal queue |
| `core/vector_store.py` | Semantic memory with embeddings |
| `core/logging_utils.py` | Structured JSON logging with secret masking |

### Model Routing (aura.config.json)

| Task | Default Model |
|------|--------------|
| code_generation | anthropic/claude-3.5-sonnet |
| planning | deepseek/deepseek-chat |
| analysis | google/gemini-2.5-flash |
| critique | deepseek/deepseek-r1-0528 |
| embedding | openai/text-embedding-3-small |
| fast | google/gemini-2.0-flash-001 |
| quality | deepseek/deepseek-r1-0528 |

---

## Agent System

### Pipeline Agents

| Agent | Phase | Purpose |
|-------|-------|---------|
| `ingest` | ingest | Gather project context + memory hints |
| `planner` | plan | Generate step-by-step implementation plan |
| `critic` | critique | Adversarial plan review |
| `synthesizer` | synthesize | Merge plan + critique into task bundle |
| `coder` | act | Generate code changes |
| `sandbox` | sandbox | Execute snippets in isolated subprocess |
| `verifier` | verify | Run tests/linters |
| `reflector` | reflect | Summarize outcomes, update skill weights |

### Specialized Agents

| Agent | Purpose |
|-------|---------|
| `innovation_swarm` | Multi-technique brainstorming (8 techniques) |
| `meta_conductor` | 5-phase innovation orchestration |
| `debugger` | Debug and error handling |
| `mcp_discovery` | MCP server discovery |
| `investigation` | Root cause analysis |
| `external_llm` | External LLM routing |

### Agent Base Class

```python
# agents/base.py
class Agent(ABC):
    name: str                  # unique identifier
    capabilities: list[str] = []  # semantic tags

    @abstractmethod
    def run(self, input_data: Dict) -> Dict:
        """Return JSON-serializable phase output."""
```

### Skills System

Skills are stateless helpers called from within agents (not pipeline phases):

| Skill | Purpose |
|-------|---------|
| `ast_analyzer` | Parse + query Python ASTs |
| `complexity_scorer` | McCabe complexity analysis |
| `test_coverage_analyzer` | Coverage analysis |
| `security_scanner` | Security scanning |
| `linter_enforcer` | Lint code |
| `doc_generator` | Auto-docstring generation |

---

## Configuration

### Key Configuration Files

| File | Purpose |
|------|---------|
| `aura.config.json` | Runtime config - models, MCP ports, connectors |
| `settings.json` | Model routing overrides, loop settings |
| `.mcp.json` | MCP server definitions |
| `pyproject.toml` | Package config, dependencies, pytest settings |

### MCP Servers (Ports)

| Server | Port | Purpose |
|--------|------|---------|
| dev_tools | 8001 | Main MCP server |
| skills | 8002 | Skills exposure |
| control | 8003 | Control plane |
| thinking | 8004 | Sequential thinking |
| agentic_loop | 8006 | Loop orchestration |
| copilot | 8007 | Copilot integration |
| sadd | 8020 | SADD operations |
| discovery | 8025 | Tool discovery |
| swarm | 8050 | Swarm runtime |

---

## Development Workflow

### Common Commands

```bash
# CLI help
python3 main.py --help
./run_aura.sh --help

# Goal management
python3 main.py goal add "Refactor goal queue" --run
python3 main.py goal once "Summarize repo" --dry-run

# Run tests
python3 -m pytest
python3 -m pytest tests/test_orchestrator.py -v

# Server mode
python3 -m aura_cli.server
```

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `AURA_ENABLE_NEW_ORCHESTRATOR` | `true` | Enable async orchestrator |
| `AURA_ENABLE_MCP_REGISTRY` | `true` | Enable typed agent registry |
| `AURA_ENABLE_SWARM` | `0` | Enable swarm runtime |
| `AURA_SKIP_CHDIR` | - | Keep current working directory |
| `AURA_TELEMETRY` | - | Enable telemetry DB |
| `AURA_REQUIRE_LOCAL_MODEL_HEALTH` | `0` | Verify local models before start |

---

## Testing

### Test Structure

| Directory | Purpose |
|-----------|---------|
| `tests/` | Main test suite (400+ files) |
| `tests/snapshots/` | JSON snapshot contracts |
| `tests/integration/` | Integration tests |
| `tests/e2e/` | End-to-end tests |
| `tests/fakes/` | Test fakes |
| `core/tests/` | Core module unit tests |

### Test Configuration (pyproject.toml)

```toml
[tool.pytest.ini_options]
pythonpath = [".", "experimental"]
testpaths = ["."]
asyncio_mode = "auto"
addopts = [
    "--strict-markers",
    "--cov=aura_cli",
    "--cov=core",
    "--cov=agents",
    "--cov=memory",
]
```

### Coverage Targets

| Module | Status |
|--------|--------|
| aura_cli | Tracked |
| core | Tracked |
| agents | Tracked |
| memory | Tracked |

---

## Dependencies

### Runtime Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.135.1 | Web framework |
| uvicorn | 0.44.0 | ASGI server |
| pydantic | 2.12.5 | Data validation |
| typer | >=0.12.0 | CLI framework |
| pyautogen | >=0.2.0,<0.11 | Multi-agent framework |
| gitpython | - | Git operations |
| rich | - | Terminal formatting |
| networkx | - | Graph operations |
| textblob | - | NLP processing |

### Development Dependencies

| Package | Purpose |
|---------|---------|
| pytest >=7.0 | Testing framework |
| pytest-cov >=4.0 | Coverage |
| ruff >=0.4 | Linting |
| bandit >=1.8 | Security scanning |
| pre-commit >=3.0 | Git hooks |

---

## Documentation

| File | Purpose |
|------|---------|
| `AGENTS.md` | Agent SDK reference |
| `CLAUDE.md` | AI Assistant detailed guide |
| `README.md` | Developer entry points |
| `COLLAB_CONTEXT.md` | Multi-agent collaboration |
| `docs/INTEGRATION_MAP.md` | Architecture overview |
| `docs/MCP_SERVERS.md` | MCP server documentation |
| `docs/WEB_UI_GUIDE.md` | Web UI documentation |

---

## File Inventory

### Core Modules (~140 files)

Key files in `core/`:
- `orchestrator.py`, `orchestrator_phases.py`, `orchestrator_verify.py`, `orchestrator_learn.py`, `orchestrator_capabilities.py`
- `async_orchestrator.py`, `async_bridge.py`, `async_optimizer.py`
- `model_adapter.py`, `model_providers.py`, `model_cache.py`
- `mcp_agent_registry.py`, `mcp_registry.py`, `mcp_client.py`, `mcp_contracts.py`
- `goal_queue.py`, `goal_archive.py`
- `file_tools.py`, `git_tools.py`
- `vector_store.py`, `vector_store_qdrant.py`
- `logging_utils.py`, `telemetry.py`
- `sadd/types.py`, `sadd/workstream_graph.py`, `sadd/sub_agent_runner.py`, `sadd/session_coordinator.py`
- `agent_sdk/controller.py`, `agent_sdk/tool_registry.py`, `agent_sdk/resilience.py`
- `policies/base.py`, `policies/sliding_window.py`, `policies/time_bound.py`, `policies/resource_bound.py`
- `metrics/collector.py`, `metrics/analytics.py`, `metrics/agent_performance.py`
- `notifications/manager.py`, `notifications/discord.py`, `notifications/slack.py`

### Agents (~45 files)

Key files in `agents/`:
- `base.py`, `registry.py`, `schemas.py`, `prompt_manager.py`
- `planner.py`, `critic.py`, `coder.py`, `verifier.py`, `reflector.py`
- `ingest.py`, `synthesizer.py`, `sandbox.py`
- `debugger.py`, `tester.py`, `scaffolder.py`
- `innovation_swarm.py`, `meta_conductor.py`, `multi_agent_workflow.py`
- `mcp_discovery_agent.py`, `investigation_agent.py`, `external_llm_agent.py`
- `self_correction_agent.py`, `monitoring_agent.py`, `telemetry_agent.py`
- `python_agent.py`, `typescript_agent.py`, `autogen_agent.py`
- `pr_review/review_agent.py`, `pr_review/fix_agent.py`

### Skills (~40 files)

Key files in `agents/skills/`:
- `base.py`, `registry.py`
- `ast_analyzer.py`, `complexity_scorer.py`
- `linter_enforcer.py`, `type_checker.py`
- `test_coverage_analyzer.py`, `security_scanner.py`
- `dependency_analyzer.py`, `doc_generator.py`
- `incremental_differ.py`, `git_history_analyzer.py`
- `refactoring_advisor.py`, `tech_debt_quantifier.py`

---

## Collaboration Notes

### Multi-Agent Guidelines

- **Coordinator**: Codex for complex cross-file changes
- **Research**: Gemini for upstream API/library investigation
- **GitHub**: Copilot for issue-to-PR workflow
- **Sidecar**: Kimi for isolated bounded tasks

See `COLLAB_CONTEXT.md` for active task board and handoff notes.

### Agent Routing

| Task Type | Recommended Agent |
|-----------|-------------------|
| Complex Python changes in `core/` | Codex |
| Research-heavy tasks | Gemini |
| GitHub PR/issue follow-up | Copilot |
| Isolated sidecar implementation | Kimi |

---

*This document provides a comprehensive overview of the AURA CLI codebase. For detailed implementation guidance, refer to `AGENTS.md` and individual module documentation.*
