# ADR-007: OpenAPI-First API Contract

**Date:** 2026-04-10
**Status:** Accepted
**Deciders:** Copilot (Sprint 4 automation)

---

## Context

AURA's REST API is defined by FastAPI route decorators. Previously:
- No machine-readable OpenAPI spec was committed to the repository
- There was no CI check to detect API drift (routes changing without docs updating)
- Consumers (MCP bridge, web-ui, external integrations) had no stable contract to code against
- Manual documentation in `docs/API.md` could diverge silently from the actual server

---

## Decision

Adopt an **OpenAPI-first contract discipline**:

1. **Export on every significant change**: `scripts/export_openapi.py` exports the live FastAPI spec to `docs/api/openapi.json`
2. **Spec committed to version control**: The generated `docs/api/openapi.json` is tracked in git as the source-of-truth contract
3. **CI validation job** (`openapi-validate` in `ci-enhanced.yml`):
   - Exports the spec fresh from the current code
   - Validates with `openapi-spec-validator`
   - Issues a warning if the committed spec differs from what the current code would generate (drift detection)
4. **Spec artifact upload**: Every CI run uploads `openapi.json` as a build artifact

The spec is **generated** (not hand-written) — FastAPI auto-generates it from type annotations and route decorators. This means:
- Type annotations on request/response models are the authoritative definition
- Changes to Pydantic models automatically propagate to the spec
- The spec is always syntactically correct (FastAPI validates it internally)

---

## Consequences

**Positive:**
- Consumers can code-generate clients from `docs/api/openapi.json`
- CI catches API drift before it reaches `main`
- `openapi-spec-validator` provides an independent correctness check
- No manual maintenance of API documentation

**Negative / Trade-offs:**
- `scripts/export_openapi.py` must be run before committing API changes — this is a new workflow step for developers (documented in `CONTRIBUTING.md`)
- The CI drift check is a **warning**, not a hard failure — avoids blocking PRs for trivial reorderings, but requires developer discipline
- Spec only covers the FastAPI server (port 8000); MCP server (port 8001) and Observability MCP (port 8030) have separate schemas not covered here

**Follow-on:**
- Add `export_openapi.py` as a pre-commit hook (optional, can be slow)
- Consider generating TypeScript/Python clients in CI from `docs/api/openapi.json`
- Extend to cover the MCP server schema (JSON-RPC format, different tooling needed)
