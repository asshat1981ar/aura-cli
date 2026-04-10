# ADR-004: Pydantic v2 Migration

**Date:** 2026-03-24
**Status:** Accepted
**Deciders:** AURA Core Team
**Tags:** dependencies, schemas, validation, performance

---

## Context

AURA's agent layer relies heavily on structured data validation through Pydantic
models defined in `agents/schemas.py`. As of sprint 1 the codebase used Pydantic
v1 semantics (`validator`, `__fields__`, `.dict()`, `.schema()`).

Pydantic v1 reached end-of-active-support in mid-2024. The v1 API is available
in Pydantic ≥ 2.0 only through the `pydantic.v1` compatibility shim, which adds
overhead and is not guaranteed to be maintained long-term.

Several factors drove the migration decision:

| Factor | Detail |
|--------|--------|
| **Performance** | Pydantic v2 core is written in Rust (`pydantic-core`); validation is 5–50× faster for complex nested models |
| **Typing accuracy** | Stricter type coercion catches bugs that v1 silently swallowed |
| **LTS alignment** | Pydantic v2 is the only version receiving new features and security fixes |
| **Dependency compatibility** | FastAPI ≥ 0.100, OpenAI SDK ≥ 1.0, and other AURA dependencies require v2 |
| **`model_validator` / `field_validator`** | Cleaner decorator API replaces `@validator` and `@root_validator` |

### Alternatives Considered

1. **Stay on Pydantic v1** — Rejected: EOL, incompatible with current FastAPI and
   OpenAI SDK versions, blocks future dependency upgrades.
2. **Use `pydantic.v1` compatibility shim** — Rejected: Adds an import-time overhead
   layer and would need removal eventually; does not address the root issue.
3. **Replace Pydantic with `dataclasses` + `cattrs`** — Rejected: Would lose JSON
   schema generation (`model_json_schema()`) used by LLM tool-call binding; higher
   migration cost with fewer benefits.

---

## Decision

Migrate all Pydantic models in `agents/schemas.py` and related modules to Pydantic
v2 semantics:

- Replace `@validator` with `@field_validator(mode="before"|"after")`.
- Replace `@root_validator` with `@model_validator(mode="before"|"after")`.
- Replace `.dict()` calls with `.model_dump()`.
- Replace `.schema()` calls with `.model_json_schema()`.
- Replace `class Config:` inner classes with `model_config = ConfigDict(...)`.
- Pin `pydantic==2.x` in `requirements.txt` (currently `pydantic==2.12.5`).

All 23 schema classes in `agents/schemas.py` (including `PlanStep`, `CoderOutput`,
`CriticIssue`, `InnovationOutput`, `InnovationSessionState`, etc.) were updated in
this migration.

---

## Consequences

### Positive

- Validation of agent output payloads is significantly faster, reducing per-cycle
  latency by an estimated 10–30 ms for deeply nested planner/coder schemas.
- Stricter coercion surfaces model mismatches earlier (at parse time rather than at
  field access time), improving debuggability.
- Aligns with FastAPI's native v2 integration — request/response model serialisation
  uses the Rust core path end-to-end.
- Enables `model_json_schema()` for LLM tool-call JSON schema binding without
  additional adapters.

### Negative / Risks

- **Breaking change**: Any external consumer of AURA's schema classes calling
  `.dict()` or `.schema()` must update to `.model_dump()` / `.model_json_schema()`.
  No known external consumers at time of writing.
- **`model_validator` semantics differ**: `mode="before"` validators receive raw
  dict input (not a partially constructed model); some validators required
  rewriting.
- Future Pydantic minor/patch upgrades may surface new deprecation warnings; pin
  to a specific minor version in CI to catch regressions.

### Follow-on Actions

- [ ] Add `pydantic` to the `dependabot` update policy with a patch-only auto-merge
  rule.
- [ ] Update developer documentation to use v2 API examples.
- [ ] Remove any remaining `from pydantic.v1 import …` imports if discovered.
