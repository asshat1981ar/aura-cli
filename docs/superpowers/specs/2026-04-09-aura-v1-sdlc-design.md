# AURA CLI v0.1.0 → v1.0 SDLC Design Specification

**Date:** April 9, 2026  
**Status:** Approved — Implementation in Progress  
**Repository:** github.com/asshat1981ar/aura-cli  
**Spec Author:** Design via Copilot + User Collaboration  

---

## 1. Problem Statement

AURA CLI is an autonomous AI development loop at Alpha v0.1.0. The system runs locally and on Android/Termux, processes natural-language goals through a 10-phase multi-agent pipeline, and applies code changes atomically. However, several production-blocking gaps prevent safe, team-scale deployment:

1. **God modules** — `aura_cli/server.py` (75 KB+) and `aura_cli/dispatch.py` (47 KB+) concentrate risk; a change in either can silently break multiple agents and surfaces.
2. **Low test coverage** — current floor is ~40%; critical paths (sandbox, atomic apply, reflector) are insufficiently tested, masking regressions during refactoring.
3. **LLM provider coupling** — no circuit breaker, cost cap, or deterministic mock harness; provider outages halt entire pipelines.
4. **Sandbox security** — generated code is not restricted from host filesystem writes or network access; production deployment without this is a critical security gap.
5. **Multi-surface drift** — CLI, Web UI, 5 MCP servers, and VS Code extension (WIP) share no formal API contract test; backend changes silently break surfaces.
6. **Missing observability** — no Prometheus metrics, no Grafana dashboards, no structured alerting.

---

## 2. Target State (v1.0)

| Dimension | Alpha (v0.1.0) | v1.0 Target |
|-----------|---------------|-------------|
| Test coverage | ~40% | ≥80% overall; ≥90% core pipeline |
| Largest source file | api_server.py ~75 KB | ≤500 lines / file |
| Sandbox isolation | subprocess only | fs + network + resource limits |
| CI gates | lint + basic pytest | lint + SAST + pip-audit + contract + E2E |
| Observability | stdout logs | Prometheus /metrics + Grafana dashboards |
| Docker | dev only | production docker-compose + healthchecks |
| Dashboard auth | none | JWT with refresh tokens |
| Documentation | partial | OpenAPI, SDK guide, deploy guide, A2A protocol |

---

## 3. Architecture: Proposed Decomposition

### 3.1 API Layer Decomposition (`aura_cli/api/`)

Replace the monolithic `server.py` with a router-per-domain package:

```
aura_cli/api/
├── app.py                # thin FastAPI composition root
├── routers/
│   ├── runs.py           # POST /run, GET /runs, GET /runs/{id}
│   ├── ws.py             # WebSocket /ws/{run_id}
│   ├── health.py         # GET /health, GET /ready, GET /metrics
│   ├── goals.py          # Goal queue CRUD
│   └── agents.py         # Agent registry and status
└── middleware/
    └── auth.py           # JWT auth middleware
```

### 3.2 Agent Handler Package (`agents/handlers/`)

Extract per-phase agent invocation from `dispatch.py` into isolated handler modules:

```
agents/handlers/
├── __init__.py
├── planner_handler.py    # Phase 3: Plan
├── coder_handler.py      # Phase 6: Act
├── debugger_handler.py   # Phase 6 retry: Debug
├── critic_handler.py     # Phase 4: Critique
├── reflector_handler.py  # Phase 10: Reflect
└── applicator_handler.py # Phase 8: Apply
```

### 3.3 Sandbox Hardening

- Filesystem isolation: restrict subprocess writes to a temporary working directory
- Network isolation: run sandbox with no network access (unshare or iptables rule)
- Resource limits: RLIMIT_CPU (30s), RLIMIT_AS (512 MB), wall-clock timeout (60s)
- Violation logging: emit structured `sandbox_violation` log event on policy breach

### 3.4 Memory Layer (5-Layer Contract)

| Layer | Component | Backend |
|-------|-----------|---------|
| L0 | In-process dict | Python dict |
| L1 | SQLite brain | `memory/brain_v2.db` |
| L2 | Vector store | SQLite + embeddings / Qdrant optional |
| L3 | Context graph | NetworkX (default), Neo4j (optional) |
| L4 | Goal queue | `memory/goal_queue.json` |
| Cache | L0/L1 acceleration | Redis (optional, required for production) |

---

## 4. Sprint Plan Summary

| Sprint | Focus | Key Deliverables | MoSCoW |
|--------|-------|-----------------|--------|
| S0 | Foundation | ADRs, CONTRIBUTING.md, detect-secrets, ruff D, bandit CI | Must |
| S1 | API Decomp | `aura_cli/api/` package, composition root, health endpoints | Must |
| S2 | Agent Decomp | `agents/handlers/` package, dispatch.py shrunk to ≤500 lines | Must |
| S3 | Security | Sandbox hardening, JWT hardening, cancel command, rollback test | Must |
| S4 | Unit Tests | Mock LLM harness, ≥90% planner/coder/applicator/reflector | Must |
| S5 | E2E + Integration | 3 E2E scenarios, WebSocket integration test, session persistence | Must |
| S6 | Model Router + Memory | Cost cap, circuit breaker, L2 planner context, Redis cache, aura history | Should |
| S7 | Dashboard UX | All 16 views wired, JWT auth, WCAG, Lighthouse ≥80 | Should |
| S8 | Production Hardening | Prometheus + Grafana, structlog, docker-compose.prod.yml, runbooks | Must |
| S9 | Integration + Docs | OpenAPI validation, coverage ≥80% verified, SDK guide, deploy guide | Must |
| S10 | Release | RC1 branch, regression run, pip-audit clean, v1.0.0 tag + GitHub Release | Must |

---

## 5. Testing Strategy

### 5.1 Pyramid

- **Unit** (per-module, ≥90% core pipeline): `tests/test_*.py`, mock LLM harness
- **Integration** (API + DB + agents): `tests/integration/`, real SQLite, mock LLM
- **E2E** (CLI + full pipeline): `tests/e2e/`, fixture project, mock LLM
- **Security** (SAST + pip-audit): bandit + pip-audit in CI
- **Contract** (OpenAPI): schemathesis in CI

### 5.2 Coverage Targets

| Module | Current | v1.0 Target |
|--------|---------|-------------|
| agents/planner.py | unknown | ≥90% |
| agents/coder.py | unknown | ≥90% |
| core/file_tools.py | unknown | ≥90% |
| agents/reflector.py | unknown | ≥90% |
| agents/sandbox.py | unknown | ≥90% |
| Overall | ~40% | ≥80% |

---

## 6. Security Design

### 6.1 JWT Auth
- Algorithm: HS256 minimum, RS256 preferred
- Secret: ≥256-bit random (`secrets.token_urlsafe(32)`)
- Expiry: ≤24 hours access token + refresh token flow
- HTTPS enforced via reverse proxy (nginx/caddy) in production

### 6.2 Sandbox Isolation
- No filesystem writes outside designated tmpdir
- No network access from sandboxed subprocess
- Resource limits: CPU 30s, memory 512 MB, wall-clock 60s
- All violations logged as structured events

### 6.3 Dependency Security
- pip-audit in CI; zero HIGH/CRITICAL CVEs gate v1.0 release
- bandit SAST in CI; zero HIGH findings gate v1.0 release

---

## 7. Deployment Architecture

### 7.1 Development
```bash
docker compose up   # starts: api (8001), web-ui (3000), redis
```

### 7.2 Production
```bash
docker compose -f docker-compose.prod.yml up  
# adds: nginx TLS termination, healthchecks, resource limits, secrets management
```

### 7.3 Observability Stack
- FastAPI `/metrics` → Prometheus scrape
- Prometheus → Grafana dashboards (`grafana/dashboards/aura.json`)
- structlog JSON → Loki / ELK log aggregation
- Alerting rules: pipeline failure rate, LLM latency spike, sandbox error rate

---

## 8. Risks and Mitigations

| Risk ID | Risk | Mitigation |
|---------|------|-----------|
| R-001 | God module complexity | Sprints 1–2 decomposition; 500-line lint gate |
| R-002 | Low test coverage | Sprint 4 Testing Sprint; per-module coverage enforcement |
| R-003 | LLM provider coupling | Mock harness + circuit breaker + cost cap (Sprint 6) |
| R-004 | Sandbox security | Sprint 3 hardening; internal audit (Sprint 9) |
| R-005 | Multi-surface drift | OpenAPI contract test in CI (Sprint 0) |
| R-006 | Contributor friction | CONTRIBUTING.md + [EXPERIMENTAL] tags (Sprint 0) |

---

## 9. Definition of Done (v1.0)

A sprint item is done when:
1. Code change merged to `main` via PR with CI green
2. Tests added/updated; no coverage regression
3. ruff and bandit clean
4. `docs/CLI_REFERENCE.md` regenerated if CLI changed
5. Snapshot tests updated if output changed

v1.0 release is done when:
- All Must Have stories pass acceptance criteria
- Overall test coverage ≥80%; core pipeline ≥90%
- Zero HIGH/CRITICAL CVEs in pip-audit
- Zero HIGH bandit findings
- docker-compose.prod.yml boots clean with health checks
- Production deployment guide complete
- GitHub Release published at tag v1.0.0

---

*This spec is the ground truth for sprint execution. Track execution via SQL todos in the active Copilot session.*
