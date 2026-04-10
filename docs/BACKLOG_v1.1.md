# AURA CLI — v1.1 Backlog

> Items deferred from the v1.0 SDLC plan. Each entry documents *why* the
> feature was not included in v1.0, what must be in place before work starts,
> a rough story-point estimate, and a priority rating.
>
> Story points use a Fibonacci-ish scale: 1 · 2 · 3 · 5 · 8 · 13.
> Priority: **High** = blocks growth / revenue, **Med** = valuable but
> workaround exists, **Low** = nice-to-have or experimental.

---

## 1. VS Code Extension

| Field            | Detail |
|------------------|--------|
| **Feature name** | AURA VS Code Extension |
| **Why deferred** | The scaffolding under `vscode-extension/` is WIP and not yet feature-complete. Stabilising the extension requires a stable CLI API surface, which was itself still evolving during v1.0. Shipping a broken or half-baked IDE integration would hurt first impressions more than the omission. |
| **Prerequisites** | Stable `aura` CLI public API (commands, JSON output format, exit codes); `vscode-extension/` test suite with ≥ 80 % coverage; VS Code Marketplace developer account and publisher token stored in repo secrets. |
| **Story points**  | 13 |
| **Priority**      | High |

### Scope
- Sidebar panel showing active AURA run status and cycle logs.
- One-click `aura run` / `aura stop` commands via Command Palette.
- Inline diff viewer for AI-proposed code changes.
- Settings UI for API keys, model selection, and swarm flags.

---

## 2. Orchestrator Hub (`experimental/orchestrator_hub`)

| Field            | Detail |
|------------------|--------|
| **Feature name** | Orchestrator Hub — multi-project agent coordination |
| **Why deferred** | Lives in `experimental/orchestrator_hub/` and has no production-grade error handling, persistence, or auth. The v1.0 focus was single-project reliability; adding a multi-project hub before the single-project loop was stable would have multiplied debugging surface area. |
| **Prerequisites** | AURA v1.0 single-project loop proven stable in CI; REST/WebSocket API spec for hub↔agent protocol (ADR required); Redis or equivalent message broker added to `docker-compose.prod.yml`; integration tests with ≥ 2 simultaneous agent sessions. |
| **Story points**  | 13 |
| **Priority**      | Med |

### Scope
- Central dashboard for monitoring N concurrent AURA sessions.
- Cross-project lesson sharing via shared `LessonStore`.
- Queue management: priority lanes, rate-limit pooling across projects.
- Role-based access control for multi-team deployments.

---

## 3. Neo4j L3 Context Graph

| Field            | Detail |
|------------------|--------|
| **Feature name** | Neo4j-backed L3 long-term context graph (optional driver) |
| **Why deferred** | The current implementation defaults to NetworkX (in-process, zero external deps). Neo4j adds ops burden (Docker service, Bolt auth, schema migrations) that was out of scope for v1.0. The pluggable `GraphBackend` interface exists in `core/` but the Neo4j adapter is not yet written. |
| **Prerequisites** | `core/graph_backend.py` abstract interface finalised; `neo4j` Python driver added to optional extras in `pyproject.toml`; integration tests using `testcontainers-python` Neo4j image; documented migration path from NetworkX export to Neo4j import. |
| **Story points**  | 8 |
| **Priority**      | Low |

### Scope
- `Neo4jGraphBackend` implementing the existing `GraphBackend` ABC.
- Cypher queries replacing NetworkX equivalents for node/edge CRUD.
- `aura graph migrate --to neo4j` command for existing users.
- Performance benchmarks vs NetworkX at 10 k / 100 k node scale.

---

## 4. Multi-Provider LLM Support

| Field            | Detail |
|------------------|--------|
| **Feature name** | Native Anthropic & OpenAI providers (beyond OpenRouter) |
| **Why deferred** | v1.0 routes all LLM calls through OpenRouter, which already covers both providers. Native SDKs were deprioritised to avoid duplicating retry logic, auth handling, and streaming adapters before the abstraction layer was stable. |
| **Prerequisites** | `core/llm_provider.py` provider protocol finalised; `anthropic` and `openai` packages added to optional extras; per-provider streaming adapters; token-cost tracking abstraction so telemetry remains provider-agnostic; secret rotation documented in `SECURITY.md`. |
| **Story points**  | 8 |
| **Priority**      | High |

### Scope
- `AnthropicProvider` with native `anthropic` SDK (streaming, tool-use, vision).
- `OpenAIProvider` with native `openai` SDK (streaming, function-calling, Assistants API).
- Provider selection via `aura.config.json` → `"llm_provider": "anthropic" | "openai" | "openrouter"`.
- Automatic fallback chain: primary → secondary provider on 5xx / rate-limit.
- Cost estimation table displayed in `aura run --dry-run`.

---

## 5. Android / Termux Production Deployment Guide

| Field            | Detail |
|------------------|--------|
| **Feature name** | Android/Termux production deployment guide and hardening |
| **Why deferred** | `aura.config.android.example.json` and `docs/LOCAL_MODELS_ANDROID.md` exist as early sketches, but Termux-specific packaging, battery/thermal throttling mitigations, and background service persistence were not tested at v1.0 scope. |
| **Prerequisites** | Reproducible Termux test environment documented (proot-distro or bare); CI matrix entry with `arm64` runner or emulator; `requirements-android.txt` with Termux-compatible wheel overrides for numpy / torch; user-facing `aura.config.android.example.json` validated against the v1.0 config schema. |
| **Story points**  | 5 |
| **Priority**      | Med |

### Scope
- Step-by-step `docs/ANDROID_DEPLOY.md` covering Termux package setup, Python 3.11 install, and first `aura run`.
- `scripts/install_android.sh` wrapper that detects Termux and installs compatible wheels.
- Guidance on Termux:Boot for persistent background execution.
- Thermal/battery-aware throttling config keys in `aura.config.json`.
- Known-limitations section: no Docker support, restricted filesystem paths.

---

## 6. SADD Parallel Workstream v2 — Dependency DAG Visualisation

| Field            | Detail |
|------------------|--------|
| **Feature name** | SADD v2: improved dependency DAG visualisation and scheduling |
| **Why deferred** | SADD v1 (`sadd run` / `sadd resume`) shipped the core parallel execution engine in v1.0. Advanced DAG rendering and dynamic re-scheduling were scoped out to keep the v1.0 surface small and avoid UI framework lock-in before the CLI stabilised. |
| **Prerequisites** | SADD v1 end-to-end tests green in CI; `WorkstreamGraph` serialisation format stable (no breaking changes planned); choice of visualisation backend agreed (Rich tree vs. Graphviz vs. Mermaid export); `sadd plan` sub-command accepted as the canonical entry point for DAG introspection. |
| **Story points**  | 8 |
| **Priority**      | Med |

### Scope
- `sadd plan --visualise` renders the dependency DAG in the terminal (Rich tree) or exports a Mermaid `.md` file.
- Colour-coded node states: pending · running · completed · failed · blocked.
- Critical-path highlighting (longest dependency chain).
- Dynamic re-scheduling: when a workstream fails, `sadd resume` recomputes unblocked dependents rather than retrying the full graph.
- `--max-parallel N` flag to cap concurrent workstreams for resource-constrained environments.

---

*Last updated: v1.0 release cut. Re-evaluate priorities at v1.1 planning session.*
