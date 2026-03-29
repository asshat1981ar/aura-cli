# SDLC Orchestrator — Botpress ADK Scaffold

Botpress Agent Development Kit (ADK) scaffold for the AURA SDLC orchestrator agent.

## Status

**Scaffold** — generated during Sprint S005 swarm run. Needs review and integration testing before production use.

## Structure

- `.adk/` — ADK type definitions and bot runtime
- `package.json` / `package-lock.json` — Node.js dependencies

## Setup

```bash
cd sdlc-orchestrator
npm install
```

## Integration

This scaffold is designed to be wired into the AURA CLI command dispatch system. See `aura_cli/cli_main.py` for the dispatch registry.
