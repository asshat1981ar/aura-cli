# AURA CLI — Project Overview

## Purpose
AURA CLI is an autonomous software development platform. It runs a multi-agent loop that takes natural-language goals, plans code changes, applies them atomically, sandboxes execution, verifies with tests/linters, and reflects on outcomes — all with persistent memory and adaptive skill weights.

## Tech Stack
- **Language:** Python 3.10+ (primary), some Node.js tooling
- **Framework:** Custom orchestration engine (no web framework — CLI-first)
- **Testing:** pytest with unittest-style test classes
- **Models:** Multi-provider LLM routing via OpenRouter (Gemini, Claude, GPT, DeepSeek, Qwen, Codex)
- **Memory:** SQLite + NetworkX graph, JSONL decision logs
- **MCP:** Model Context Protocol servers for tool integration
- **n8n:** Workflow automation (14 workflows for pipeline orchestration)
- **Config:** JSON-based (aura.config.json, settings.json)

## Entry Points
- `main.py` — Primary CLI shim, delegates to `aura_cli.cli_main:main()`
- `run_aura.sh` — Bash convenience wrapper

## Key Directories
```
aura_cli/     — CLI layer (commands, dispatch, options, TUI)
core/         — Orchestration engine (orchestrator, model adapter, SADD, policies)
agents/       — Specialized agents (planner, critic, coder, debugger, tester, etc.)
memory/       — State persistence (brain, store, embedding)
tools/        — MCP servers
tests/        — Test suite (4,700+ tests)
n8n-workflows/ — n8n workflow JSON files (WF-0 through WF-8)
```
