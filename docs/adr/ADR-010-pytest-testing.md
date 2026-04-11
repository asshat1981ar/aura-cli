# ADR-010: pytest Testing Framework

**Date:** 2026-04-10  
**Status:** Accepted  
**Deciders:** AURA Core Team  

## Context

AURA CLI needed a comprehensive testing framework to ensure:

1. Reliable test execution across different environments
2. Support for async code testing
3. Good integration with coverage reporting
4. Fixtures and parameterized testing
5. Clear test organization and discovery
6. CI/CD integration
7. Plugin ecosystem for specialized testing needs

The main contenders were:
- **unittest** (stdlib) — Built-in, no dependencies, but verbose
- **pytest** — Industry standard, rich plugin ecosystem
- **nose2** — unittest extension, less popular than pytest
- **green** — Colorful output, but less feature-rich

## Decision

We chose **pytest** as our testing framework.

### Rationale

1. **Assert Rewriting**: Clean, readable assertions
   ```python
   # pytest automatically shows detailed diffs
   def test_config():
       config = load_config()
       assert config.log_level == "INFO"  # Clear failure message if wrong
   ```

2. **Fixtures**: Powerful dependency injection for tests
   ```python
   @pytest.fixture
   def temp_config():
       with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
           json.dump({"log_level": "DEBUG"}, f)
           yield f.name
   
   def test_load_config(temp_config):
       config = load_config(temp_config)
       assert config.log_level == "DEBUG"
   ```

3. **Parameterized Tests**: Run same test with multiple inputs
   ```python
   @pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARN", "ERROR"])
   def test_valid_log_levels(log_level):
       config = AuraConfig(log_level=log_level)
       assert config.log_level == log_level
   ```

4. **Async Support**: Native support for async/await
   ```python
   @pytest.mark.asyncio
   async def test_async_orchestrator():
       result = await orchestrator.run_goal("test")
       assert result.success
   ```

5. **Plugin Ecosystem**:
   - `pytest-cov` — Coverage reporting
   - `pytest-asyncio` — Async test support
   - `pytest-timeout` — Prevent hanging tests
   - `pytest-xdist` — Parallel test execution

6. **Markers**: Categorize and select tests
   ```python
   @pytest.mark.slow
   @pytest.mark.integration
   def test_full_pipeline():
       ...
   ```

## Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
pythonpath = [".", "experimental"]
testpaths = ["."]
asyncio_mode = "auto"
timeout = 30
timeout_method = "thread"
addopts = [
    "--strict-markers",
    "-q",
    "--cov=aura_cli",
    "--cov=core",
    "--cov=agents",
    "--cov=memory",
    "--cov-report=term-missing",
]
markers = [
    "unit: marks pure unit tests",
    "integration: marks integration tests",
    "slow: marks tests as slow (>10s)",
    "e2e: marks end-to-end tests",
    "security: marks security-focused tests",
]
```

## Test Organization

```
tests/
├── unit/              # Fast, isolated tests
│   ├── test_config.py
│   ├── test_sanitizer.py
│   └── test_auth.py
├── integration/       # Component interaction tests
│   ├── test_server_api.py
│   └── test_goal_queue.py
├── e2e/              # Full workflow tests
│   └── test_pipeline.py
├── fixtures/         # Shared test data
│   └── sample_configs/
└── helpers/          # Test utilities
    ├── fixtures.py
    └── mocks.py
```

## Testing Strategy

| Test Type | Target | Tools | Execution |
|-----------|--------|-------|-----------|
| Unit | Individual functions | pytest, mocks | Every commit |
| Integration | Component interactions | pytest, test DB | PR merge |
| E2E | Full workflows | pytest, Docker | Nightly |
| Security | Vulnerability scanning | bandit, safety | PR + Nightly |
| Performance | Benchmarks | pytest-benchmark | Weekly |

## Consequences

### Positive

- Expressive, readable test code
- Rich assertion failure messages
- Excellent async testing support
- Comprehensive coverage reporting
- Easy test selection and organization
- Strong IDE integration

### Negative

- Additional development dependencies
- Learning curve for advanced features
- Some plugins may have compatibility issues
- Test collection time can grow with codebase size

## Best Practices

1. **Use fixtures** for setup/teardown
2. **Parametrize** similar test cases
3. **Mark slow tests** for selective execution
4. **Use tmp_path** for temporary files
5. **Mock external services** in unit tests
6. **Keep tests fast** — use integration tests sparingly

## References

- [pytest Documentation](https://docs.pytest.org/)
- [Testing Guide](https://github.com/asshat1981ar/aura-cli/tree/main/tests)
- [Coverage Configuration](https://github.com/asshat1981ar/aura-cli/blob/main/pyproject.toml)
