# Contributing to AURA CLI

Welcome to AURA CLI — an autonomous development platform with a 10-phase
pipeline (ingest → plan → critique → code → apply → verify → reflect → adapt → evolve → archive).

---

## Development Setup

```bash
# Clone the repository
git clone <repo-url>
cd aura-cli

# Create virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate

# Install runtime + dev dependencies
pip install -r requirements.txt
pip install pytest pytest-timeout pytest-cov ruff pre-commit

# Install pre-commit hooks
pre-commit install

# Set required env vars for testing
export AURA_SKIP_CHDIR=1    # Keeps working directory stable during tests
export AURA_TEST_MODE=1     # Enables test-only bypasses (e.g. JWT algorithm=none)

# Verify setup
python3 main.py --help
python3 -m ruff check .
```

---

## Running Tests

> ⚠️ **CRITICAL**: Never run `pytest tests/` without a timeout.
> Some test files hang indefinitely (>2 minutes). Always use the targeted safe suite below.

### Fast Regression Suite (~4–6 seconds)

```bash
python3 -m pytest \
  tests/test_auth.py \
  tests/test_jwt_hardening.py \
  tests/test_server_api.py \
  tests/test_cli_exit_codes.py \
  tests/test_sanitizer.py \
  tests/test_correlation.py \
  tests/test_config_schema.py \
  tests/test_redis_cache.py \
  tests/test_sandbox_violations.py \
  tests/test_sandbox_unit.py \
  tests/test_e2e_sandbox_retry.py \
  -v --timeout=30 --no-cov
```

### With Coverage

```bash
python3 -m pytest tests/test_auth.py tests/test_sanitizer.py \
  -v --timeout=30 \
  --cov=aura_cli --cov=core --cov=agents --cov=memory \
  --cov-report=term-missing
```

### Run a Single Test

```bash
python3 -m pytest tests/test_auth.py::test_create_access_token -v --no-cov
```

### Triage All Test Files (finds hanging tests)

```bash
python3 scripts/triage_tests.py --timeout 30
# Output: reports/test-triage.json, reports/safe-tests.txt
```

---

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `aura_cli/` | CLI interface + FastAPI server (`server.py`, routers, middleware) |
| `core/` | Orchestration engine (~120 modules: auth, config, sanitizer, correlation…) |
| `agents/` | Specialized pipeline agents (ingest, planner, coder, sandbox, applicator…) |
| `memory/` | Persistence layer (SQLite, JSONL decision log, Redis cache) |
| `tools/` | MCP servers (mcp_server.py, sadd_mcp_server.py, observability_mcp.py) |
| `tests/` | 277 test files (unit, integration, e2e) |
| `scripts/` | Development utilities (triage_tests.py, validate_config.py, find_coverage_gaps.py) |
| `docs/` | Architecture documentation, ADRs, security audits |

---

## Entry Points

| Command | Purpose |
|---------|---------|
| `python3 main.py` | CLI via `aura_cli.cli_main:main()` |
| `python3 -m aura_cli.server` | Start canonical dev-tools FastAPI server (port 8001) |
| `python3 tools/mcp_server.py` | Start legacy MCP compatibility server |
| `python3 tools/sadd_mcp_server.py` | Start SADD MCP server (port 8020) |
| `python3 tools/observability_mcp.py` | Start Observability MCP (port 8030) |
| `bash run_aura.sh` | Convenience wrapper for CLI |

---

## Commit Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): brief description

[optional body]

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```

### Types

| Type | When |
|------|------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `test` | Adding or fixing tests |
| `docs` | Documentation changes |
| `refactor` | Code restructuring (no behavior change) |
| `perf` | Performance improvement |
| `ci` | CI/CD changes |
| `chore` | Build system, dependencies, tooling |

### Scopes

Use sprint numbers (`s1`–`s10`) for sprint work, or module names (`auth`, `sandbox`, `server`, `memory`, etc.) for targeted changes.

### Examples

```
feat(core): add rate limiting to model adapter
fix(sandbox): handle RLIMIT_AS gracefully on macOS
test(auth): add JWT algorithm confusion attack tests
docs(api): document authentication flow with sequence diagram
ci: add weekly dependency audit workflow
```

---

## Pull Request Process

1. Create a feature branch from `main`: `git checkout -b feat/your-feature`
2. Make changes with tests (maintain or increase coverage)
3. Ensure pre-commit hooks pass: `pre-commit run --all-files`
4. Run the safe test suite and verify all 165+ tests pass
5. Update `CHANGELOG.md` for user-facing changes
6. Submit PR with a clear description linking to any related issues

---

## Security Policy

| Rule | Detail |
|------|--------|
| ❌ `algorithm="none"` | NEVER use in production code. Use `tests/fakes/fake_auth.py` for test auth bypass |
| ❌ `aura_auth.db` | NEVER commit — contains JWT revocation secrets |
| ❌ `.env` files | NEVER commit — use environment variables |
| ✅ Input validation | ALWAYS validate through `core/sanitizer.py` |
| ✅ Sandbox code | ALWAYS run untrusted code through `agents/sandbox.py` |
| ✅ Secret rotation | Rotate `AURA_AUTH_SECRET_KEY` if compromised |

To report security vulnerabilities: open a private GitHub issue or contact the maintainers directly.

---

## Key Gotchas

1. **`pytest tests/` hangs** — always use the targeted safe test file list with `--timeout=30`
2. **prometheus_client changes metrics behavior** — monkeypatch `_PROMETHEUS_AVAILABLE = False` in tests that check JSON metrics
3. **`SandboxResult` requires keyword args** — uses `@dataclass(kw_only=True)`; never use positional args
4. **`python-jose[cryptography]`** must be installed for JWT tests (not just `python-jose`)
5. **`AURA_SKIP_CHDIR=1`** is required to keep working directory stable during tests
6. **`AURA_TEST_MODE=1`** must be set for tests that use `algorithm="none"` JWT bypass
7. **Coverage scope** — always use `--cov=aura_cli --cov=core --cov=agents --cov=memory`, not `--cov=.` (inflates denominator with tools/)
8. **SQLite WAL mode** — all SQLite connections now use WAL; `:memory:` DBs in tests are unaffected

---

## Development Utilities

```bash
# Validate configuration files
python3 scripts/validate_config.py

# Find highest-impact modules to add tests for
python3 scripts/find_coverage_gaps.py --top 10

# Triage all test files (find hanging ones)
python3 scripts/triage_tests.py --timeout 30

# Generate CLI reference docs
python3 scripts/generate_cli_reference.py

# Check CLI reference is current (CI check)
python3 scripts/generate_cli_reference.py --check
```
