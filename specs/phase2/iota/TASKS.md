# IOTA Implementation Tasks

> **Agent**: IOTA  
> **Status**: 🔵 READY TO START  
> **Estimated Duration**: 3 days

---

## Day 1: Core Engine (Monday)

### Morning: Project Setup
- [ ] Create directory structure: `aura/error_resolution/`
- [ ] Create `__init__.py` with public API exports
- [ ] Create `types.py` with type definitions
- [ ] Set up test directory: `tests/unit/error_resolution/`

### Afternoon: Cache Implementation
- [ ] Implement L1 in-memory LRU cache
- [ ] Implement L2 SQLite disk cache
- [ ] Create `cache.py` with `FourLayerCache` class
- [ ] Write cache tests (4 tests)

**Deliverable**: Cache layer functional with tests passing

---

## Day 2: AI Providers (Tuesday)

### Morning: Provider Abstraction
- [ ] Create `providers.py` with `AIProvider` ABC
- [ ] Implement `OpenAIProvider` with API integration
- [ ] Add retry logic from Phase 1 (`core/retry.py`)
- [ ] Write OpenAI provider tests (4 tests)

### Afternoon: Ollama Integration
- [ ] Implement `OllamaProvider` for local LLM
- [ ] Create `ProviderRegistry` for provider management
- [ ] Add provider configuration loading
- [ ] Write Ollama provider tests (3 tests)

**Deliverable**: Both providers working with configurable registry

---

## Day 3: Integration & Polish (Wednesday)

### Morning: Resolution Engine
- [ ] Implement `ErrorResolutionEngine` class
- [ ] Add known fixes registry (`known_fixes.py`)
- [ ] Implement response parser (`parser.py`)
- [ ] Create safety checker (`safety.py`)
- [ ] Write engine tests (5 tests)

### Afternoon: Error Presenter Integration
- [ ] Extend `error_presenter.py` with resolution
- [ ] Add configuration schema for error resolution
- [ ] Create CLI commands for cache management
- [ ] Write integration tests (4 tests)
- [ ] Update documentation

**Deliverable**: End-to-end error resolution working

---

## Test Checklist

### Unit Tests (15 total)
- [ ] `test_cache_l1_memory.py` - 3 tests
- [ ] `test_cache_l2_disk.py` - 3 tests
- [ ] `test_cache_integration.py` - 2 tests
- [ ] `test_openai_provider.py` - 4 tests
- [ ] `test_ollama_provider.py` - 3 tests
- [ ] `test_provider_registry.py` - 2 tests
- [ ] `test_safety_checker.py` - 4 tests
- [ ] `test_resolution_engine.py` - 5 tests
- [ ] `test_known_fixes.py` - 2 tests
- [ ] `test_parser.py` - 3 tests

### Integration Tests (5 total)
- [ ] `test_end_to_end_resolution.py` - 2 tests
- [ ] `test_error_presenter_integration.py` - 2 tests
- [ ] `test_configuration.py` - 1 test

---

## Files to Create

```
aura/error_resolution/
├── __init__.py
├── types.py
├── cache.py
├── providers.py
├── known_fixes.py
├── parser.py
├── safety.py
└── engine.py

tests/unit/error_resolution/
├── __init__.py
├── test_cache.py
├── test_providers.py
├── test_safety.py
├── test_engine.py
└── test_parser.py

tests/integration/error_resolution/
├── __init__.py
├── test_end_to_end.py
└── test_integration.py
```

---

## Dependencies

### From Phase 1
- `core/retry.py` - for provider API retries
- `core/container.py` - for dependency injection
- `aura_cli/error_presenter.py` - for UI integration
- `core/config_schema.py` - for configuration

### External
- `openai>=1.0.0` - OpenAI API client
- `aiohttp>=3.8.0` - for Ollama HTTP calls
- `diskcache>=5.0` - for L2 disk cache (optional)

---

## Daily Standup Notes

### Day 1 (Monday)
**Completed:**
- 

**In Progress:**
- 

**Blockers:**
- None

### Day 2 (Tuesday)
**Completed:**
- 

**In Progress:**
- 

**Blockers:**
- None

### Day 3 (Wednesday)
**Completed:**
- 

**In Progress:**
- 

**Blockers:**
- None

---

## Definition of Done

- [ ] All 15+ unit tests passing
- [ ] All 5+ integration tests passing
- [ ] Code coverage ≥80% for new modules
- [ ] Type hints complete
- [ ] Docstrings complete
- [ ] No linting errors
- [ ] CI passing
- [ ] Documentation updated

---

*Tasks created: 2026-04-10*  
*Start: Upon assignment*
