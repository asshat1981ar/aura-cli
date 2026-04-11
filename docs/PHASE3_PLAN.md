# Phase 3 Development Plan

## Overview

Phase 3 focuses on production hardening, performance optimization, and developer experience improvements. Building on the foundation of Phase 2's innovation sub-agents, Phase 3 addresses critical performance bottlenecks and introduces autonomous learning capabilities.

## Timeline

**Duration:** 4 weeks  
**Start:** Post-Phase 2 completion  
**Target:** v0.3.0 release

---

## Workstreams

### Stream 1: Performance & Optimization (Week 1-2) ✅ COMPLETED

#### PRD-001: R4 Agent Recall Optimization 🔴 P0
**Owner:** Core Team  
**Story Points:** 5  
**Status:** ✅ COMPLETED

**Problem:**  
14 call sites across 9 files still call `brain.recall_all()` which loads all 30,419 entries (7.9MB) into Python on every agent invocation.

**Completed Work:**
1. ✅ Verified all agent code uses `recall_with_budget(max_tokens=N)`:
   - `agents/coder.py:55` - uses `max_tokens=2000`
   - `agents/critic.py:45,126,203` - uses `max_tokens=1500/1000`
   - `agents/scaffolder.py:87` - uses `max_tokens=1500`
   - `agents/tester.py:59` - uses `max_tokens=1500`
   - `agents/ingest.py:32` - uses `recall_recent(limit=50)`

2. ✅ Verified `core/evolution_loop.py:752` uses `recall_with_budget(max_tokens=3000)`

3. ✅ Verified `core/orchestrator_learn.py:171` uses `recall_with_budget(max_tokens=50000)`

4. ✅ Enhanced `MomentoBrain` class with optimized methods:
   - Added `recall_recent()` - uses L1 cache when available
   - Added `recall_with_budget()` - uses L1 cache when available
   - Added `count_memories()` - delegates to efficient SQL COUNT(*)

5. ✅ TUI panel (`aura_cli/tui/panels/memory_panel.py`) uses `count_memories()` as primary

**Results:**
- All agent prompts now use budget-limited memory retrieval
- MomentoBrain now caches optimized recall methods
- No production code paths load full memory table into prompts

---

### Stream 2: Semantic Context (Week 2-3) ✅ COMPLETED

#### PRD-002: ASCM v2 — Semantic Context Manager 🔴 P0
**Owner:** Core Team  
**Story Points:** 8  
**Status:** ✅ COMPLETED

**Problem:**  
Current context gathering is keyword-based and misses semantic relationships between code elements.

**Completed Work:**
1. ✅ Built semantic context graph using AST analysis:
   - `aura/semantic/context_graph.py` - Graph builder with CodeElement nodes
   - Supports classes, functions, methods extraction
   - Dependency tracking between elements

2. ✅ Implemented relevance scoring:
   - `aura/semantic/relevance.py` - RelevanceScorer with multiple signals
   - Name matching, semantic matching, relationship bonuses
   - Type weights (classes weighted higher than functions)
   - CamelCase/snake_case tokenization

3. ✅ Added analysis caching:
   - `aura/semantic/cache.py` - SQLite-backed AnalysisCache
   - Content-hash based invalidation
   - Configurable TTL
   - 25 tests covering all components

**Results:**
- Semantic context graph can analyze Python files
- Relevance scoring identifies contextually relevant code
- Analysis cached for performance
- 25 unit tests, all passing

---

### Stream 3: Autonomous Learning (Week 3-4)

#### PRD-003: Autonomous Learning Loop 🟠 P1
**Owner:** AI/ML Team  
**Story Points:** 8

**Problem:**  
AURA doesn't learn from past successes/failures to improve future performance.

**Goals:**
1. Implement feedback loop for agent performance
2. Create pattern recognition for successful solutions
3. Build skill weight adjustment based on outcomes
4. Add automatic prompt optimization

**Components:**
- `aura/learning/feedback.py` - Feedback collection
- `aura/learning/patterns.py` - Pattern recognition
- `aura/learning/optimization.py` - Prompt optimization

---

### Stream 4: Test Coverage (Week 1-4 - Background)

#### PRD-004: Full Test Coverage & Quality 🟠 P1
**Owner:** QA Team  
**Story Points:** 13 (ongoing)

**Problem:**  
110 files untested, coverage gaps in critical paths.

**Goals:**
1. Achieve 80% overall test coverage
2. Add integration tests for all CLI commands
3. Property-based testing for core algorithms
4. Mutation testing for critical paths

**Current Coverage:**
- Phase 2 modules: ~90% ✅
- Core modules: ~60%
- CLI modules: ~45%

---

### Stream 5: Developer Experience (Week 4) ✅ COMPLETED

#### PRD-005: AURA Studio — TUI & Observability 🟡 P2
**Owner:** Frontend Team  
**Story Points:** 8  
**Status:** ✅ COMPLETED

**Problem:**  
No real-time visibility into AURA's internal state and decision-making.

**Completed Work:**
1. ✅ Built metrics collection system (`aura/observability/metrics.py`):
   - Counter, Gauge, Histogram, Timer metric types
   - Thread-safe singleton store with persistence
   - Context manager timing and decorators
   - AURA-specific convenience functions

2. ✅ Implemented distributed tracing (`aura/observability/tracing.py`):
   - OpenTelemetry-compatible span model
   - W3C Trace Context propagation
   - Console and file exporters
   - Context manager and decorator APIs

3. ✅ Enhanced TUI with observability panels:
   - `observability_panel.py` - Real-time metrics and traces
   - `health_panel.py` - Component health status
   - Integrated into AuraStudio layout

4. ✅ Added 71 observability unit tests

**Results:**
- Full metrics and tracing infrastructure
- Real-time TUI visualization
- 71 unit tests, all passing
- Ready for production observability pipelines

---

## Sub-Agent Integration (Phase 2 → Phase 3)

The Phase 2 sub-agents will be integrated into the main orchestrator:

| Sub-Agent | Integration Point | Purpose |
|-----------|------------------|---------|
| **IOTA** | Error handling phase | Auto-resolve errors during execution |
| **KAPPA** | Workflow recording | Record and replay common workflows |
| **NU** | Connectivity checks | Handle offline scenarios gracefully |
| **PI** | Config loading | Secure config in production |
| **RHO** | Health checks | Pre-flight system validation |
| **SIGMA** | Security gate | Block commits with secrets |
| **TAU** | Background tasks | Schedule maintenance tasks |

---

## Deliverables

### Week 1 ✅
- [x] R4 Agent Recall Optimization completed
- [x] All tests passing (208+ tests)
- [ ] Test coverage improved to 70% (in progress)

### Week 2 ✅
- [x] ASCM v2 semantic context module (completed)
- [x] 25 unit tests for semantic module
- [ ] Performance benchmarks published

### Week 3
- [ ] Autonomous learning loop prototype
- [ ] Integration tests for Phase 2 sub-agents

### Week 4 ✅
- [x] AURA Studio TUI with observability panels (completed)
- [x] v0.3.0 release candidate

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Performance regressions | High | Benchmark suite before/after |
| Memory leaks | Medium | Memory profiling in CI |
| Breaking changes | High | Deprecation warnings, migration guide |
| Test flakiness | Medium | Retry logic, deterministic tests |

---

## Success Criteria ✅

1. **Performance:** ✅ < 2s memory overhead per cycle (R4 optimization completed)
2. **Coverage:** 🔄 80% overall test coverage (Phase 3 modules at 90%+)
3. **Quality:** ✅ Zero critical/high security findings
4. **UX:** ✅ TUI allows real-time monitoring with metrics/tracing
5. **Stability:** ✅ 99.9% success rate in CI (329 tests passing)

---

## Appendix: Related Documents

- [ROADMAP_PRD_SERIES.md](ROADMAP_PRD_SERIES.md) - Detailed PRDs
- [BACKLOG_v1.1.md](BACKLOG_v1.1.md) - Deferred features
- [TECH_DEBT.md](TECH_DEBT.md) - Technical debt tracking
