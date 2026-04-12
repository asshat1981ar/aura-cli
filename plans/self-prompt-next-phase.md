# Self-Prompt: Next Phase Selection

## Current State (Post-Phase 3)

**Completed:**
- ✅ Phase 3: Production Hardening & Learning Loop (PRD-001 through PRD-005)
- ✅ 329 tests passing in `tests/aura/` modules
- ✅ Coverage gaps resolved (0 critical/high gaps)
- ✅ 8 Phase 2 sub-agents implemented and tested (IOTA, KAPPA, NU, PI, RHO, SIGMA, TAU)

**Module Inventory:**
- `aura/encryption/` - 13 tests ✅
- `aura/health/` - 25 tests ✅
- `aura/learning/` - 25 tests ✅
- `aura/observability/` - 71 tests ✅
- `aura/scheduler/` - 29 tests ✅
- `aura/security/` - 24 tests ✅
- `aura/semantic/` - 25 tests ✅
- `aura/shell/` - 29 tests ✅

---

## Next Phase Options

### Option 1: Sprint S006 — Production Hardening & Scale 🔥 RECOMMENDED

**Why this matters:** AURA has the features; now it needs production readiness for real deployments.

**Components:**
1. **Docker Compose Production Setup** (`docker-compose.prod.yml`)
   - Multi-service orchestration (app, Redis, PostgreSQL)
   - Health checks and restart policies
   - Volume management for persistence

2. **Kubernetes Deployment Manifests** (`k8s/`)
   - Deployments, Services, ConfigMaps
   - Horizontal Pod Autoscaler
   - Ingress configuration

3. **Redis Caching Layer** (`aura/cache/`)
   - Distributed caching for semantic analysis
   - Session store for TUI state
   - Rate limiting counters

4. **Async Task Queue** (`aura/tasks/`)
   - Celery or RQ for background job processing
   - Agent execution queue with priority
   - Result backends

5. **OIDC/OAuth2 Authentication** (`aura/auth/`)
   - JWT token validation
   - Role-based access control (RBAC)
   - API key management

**Deliverables:**
- Production-ready deployment configs
- 50+ new tests for production modules
- Load testing benchmarks
- Security audit passing

**Effort:** 8-10 story points (2-3 weeks)

---

### Option 2: Phase 2 Sub-Agent Integration

**Why this matters:** 8 sub-agents exist but aren't wired into the main orchestrator.

**Integration Points:**
| Sub-Agent | Target Phase | Purpose |
|-----------|--------------|---------|
| IOTA | Error handling | Auto-resolve execution errors |
| KAPPA | Workflow recording | Record/replay common workflows |
| NU | Connectivity | Handle offline scenarios |
| PI | Config loading | Secure production config |
| RHO | Health checks | Pre-flight validation |
| SIGMA | Security gate | Block commits with secrets |
| TAU | Background tasks | Schedule maintenance |

**Deliverables:**
- Sub-agent registry in orchestrator
- Phase-specific agent selection
- 40+ integration tests
- Documentation updates

**Effort:** 5-8 story points (1-2 weeks)

---

### Option 3: BEADS-Orchestrator Convergence

**Why this matters:** High-value autonomous runs need PRD-backed, decision-auditable gating.

**Components:**
1. **BEADS Bridge** (`core/beads_bridge.py`)
   - Python ↔ Node.js JSON contract
   - PRD context assembly
   - Decision validation

2. **Orchestrator Gate**
   - Pre-plan phase BEADS check
   - Allow/Block/Revise decision handling
   - Cycle persistence integration

3. **Operator Surfaces**
   - CLI status for BEADS decisions
   - TUI visibility into decision layer
   - SSE for real-time updates

**Deliverables:**
- BEADS bridge with stable contract
- Gated orchestrator runs
- Decision audit trail
- 30+ tests

**Effort:** 8 story points (2 weeks)

---

### Option 4: Comprehensive Test Coverage Drive

**Why this matters:** Core modules at ~60%, CLI at ~45% — need 80% for confidence.

**Target Modules:**
- `core/` modules (orchestrator, workflow engine, file_tools)
- `aura_cli/` commands and dispatch
- Integration tests for full goal lifecycle

**Deliverables:**
- 100+ new tests
- Property-based testing for algorithms
- Integration test suite
- Coverage reporting in CI

**Effort:** 13 story points (3-4 weeks, can parallelize)

---

## Recommendation

**Start with Option 1 (Sprint S006)** because:
1. Phase 3's observability creates the foundation for production monitoring
2. Redis caching addresses the memory performance bottleneck
3. Docker/K8s configs enable real-world deployment feedback
4. OAuth2 secures the API for multi-user scenarios

**Sequence:**
1. Week 1: Docker Compose + Redis caching
2. Week 2: Kubernetes manifests + OAuth2
3. Week 3: Sub-agent integration (Option 2)
4. Week 4: BEADS convergence (Option 3)

---

## Immediate Next Action

If proceeding with **Option 1 (Sprint S006)**:

```bash
# 1. Create production Docker setup
mkdir -p docker/prod k8s/

# 2. Initialize Redis caching module
mkdir -p aura/cache

# 3. Initialize auth module  
mkdir -p aura/auth

# 4. Start with docker-compose.prod.yml
```

**Say "next" to proceed with Option 1, or specify Option 2/3/4.**
