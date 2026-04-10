# Architecture Decision Records — Index

This document lists all ADRs in this repository in order.
New ADRs should be added here when created.

---

| # | File | Title | Status | Date | Summary |
|---|------|-------|--------|------|---------|
| 0001 | [0001-cli-first-product-boundary.md](0001-cli-first-product-boundary.md) | CLI-First Product Boundary | Accepted | 2025-01-01 | AURA ships as a CLI-first product; `aura` console script is the canonical entrypoint; other surfaces (VS Code extension, `orchestrator_hub`) are experimental. |
| 001 | [ADR-001-api-server-decomposition.md](ADR-001-api-server-decomposition.md) | api_server.py Decomposition Plan | Proposed | 2025-01-01 | Extract the 703-line monolithic `server.py` into a modular `aura_cli/api/` package with clean separation of FastAPI app, auth, routing, and WebSocket concerns. |
| 002 | [ADR-002-redis-optionality.md](ADR-002-redis-optionality.md) | Redis Optionality Contract | Accepted | 2025-01-01 | Redis (L0 cache) is an optional performance tier; AURA must function correctly when `REDIS_URL` is absent, falling back to L1 SQLite cache without errors. |
| 003 | [ADR-003-agent-handlers-package.md](ADR-003-agent-handlers-package.md) | Agent Handlers Package | Accepted | 2025-01-01 | Agent handler logic is encapsulated in a dedicated `agents/` package; each agent (Planner, Coder, Critic, Sandbox) is a self-contained class with a standard interface. |
| 004 | [ADR-004-pydantic-v2-migration.md](ADR-004-pydantic-v2-migration.md) | Pydantic v2 Migration | Accepted | 2026-03-24 | Migrate all agent schemas from Pydantic v1 to Pydantic v2 (2.x) for improved performance, stricter validation, and long-term support. |
| 005 | [ADR-005-rate-limiting-token-bucket.md](ADR-005-rate-limiting-token-bucket.md) | Rate Limiting — Token Bucket Algorithm | Accepted | 2026-04-09 | Implement per-endpoint, per-user rate limiting using the token-bucket algorithm as in-process FastAPI middleware; Redis-backed distributed limiter deferred to v1.1. |
| 006 | [ADR-006-multi-stage-docker-build.md](ADR-006-multi-stage-docker-build.md) | Multi-Stage Docker Build | Accepted | 2026-04-10 | Switch to two-stage Dockerfile (builder + runtime) to exclude build tools from production image; add .dockerignore; align entry point to uvicorn aura_cli.server:app on port 8000. |
| 007 | [ADR-007-openapi-first-contract.md](ADR-007-openapi-first-contract.md) | OpenAPI-First API Contract | Accepted | 2026-04-10 | Commit generated docs/api/openapi.json as the machine-readable API contract; validate and drift-check in CI via openapi-validate job in ci-enhanced.yml. |

---

## Status Definitions

| Status | Meaning |
|--------|---------|
| **Proposed** | Under discussion; not yet implemented |
| **Accepted** | Decision made and implemented |
| **Superseded** | Replaced by a later ADR (link provided in file) |
| **Deprecated** | No longer relevant; kept for historical record |

---

## Adding a New ADR

1. Create a file named `ADR-NNN-short-title.md` in this directory.
2. Use the template below.
3. Add a row to the table above.

```markdown
# ADR-NNN: Title

**Date:** YYYY-MM-DD
**Status:** Proposed | Accepted | Superseded by ADR-XXX
**Deciders:** Team / individuals involved

## Context

Why is this decision needed?

## Decision

What was decided?

## Consequences

What are the trade-offs and follow-on actions?
```
