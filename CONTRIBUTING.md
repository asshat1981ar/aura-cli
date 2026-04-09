# Contributing to AURA CLI

Thank you for your interest in contributing to AURA CLI! This guide covers everything you need to go from zero to an accepted pull request.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Required Tools](#required-tools)
- [Branch Naming](#branch-naming)
- [Commit Message Format](#commit-message-format)
- [Running Tests](#running-tests)
- [Running Lint](#running-lint)
- [Running Security Checks](#running-security-checks)
- [Pull Request Checklist](#pull-request-checklist)
- [EXPERIMENTAL Feature Convention](#experimental-feature-convention)
- [Agent Development](#agent-development)
- [Reporting Issues](#reporting-issues)
- [Code of Conduct](#code-of-conduct)

---

## Development Setup

### Prerequisites

| Tool | Minimum version | Install |
|---|---|---|
| Python | 3.10 (3.11 recommended) | [python.org](https://www.python.org/downloads/) |
| Git | 2.30+ | system package manager |
| pre-commit | 3.0+ | `pip install pre-commit` |
| ruff | 0.4+ | included in `.[dev]` |
| bandit | 1.7+ | included in `.[dev]` |

### Step-by-step setup

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/aura-cli.git
cd aura-cli

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate       # Linux / macOS
# venv\Scripts\activate        # Windows

# 3. Install the package in editable mode with all dev dependencies
pip install -e ".[dev]"

# 4. (Optional) Install Redis and GitHub extras
pip install -e ".[dev,redis,github]"

# 5. Install pre-commit hooks
pre-commit install

# 6. Copy the environment template and fill in your values
cp .env.example .env
# Edit .env — at minimum set OPENROUTER_API_KEY or ANTHROPIC_API_KEY
```

> **Termux / Android:** Follow `docs/LOCAL_MODELS_ANDROID.md` for the Termux-specific setup path. The `python3 -m venv` step is identical; `pip install -e ".[dev]"` works without modification.

### Verifying your setup

```bash
# Should print the AURA CLI help text
python3 -m aura_cli --help

# Should start the dev-tools server on port 8001
AGENT_API_TOKEN=test uvicorn aura_cli.server:app --port 8001

# Health check
curl http://localhost:8001/health
```

---

## Required Tools

These tools are checked by the pre-commit config and CI:

| Tool | Purpose | Config file |
|---|---|---|
| **ruff** | Linting + formatting | `ruff.toml` |
| **bandit** | Security static analysis | inline via `bandit -r` |
| **pytest** | Unit + integration tests | `pyproject.toml` `[tool.pytest]` |
| **pre-commit** | Git hook runner | `.pre-commit-config.yaml` |

Install all at once:
```bash
pip install -e ".[dev]"
```

---

## Branch Naming

Use one of the following prefixes:

| Pattern | Use case | Example |
|---|---|---|
| `feature/<issue-number>-short-description` | New features | `feature/42-redis-goal-queue` |
| `hotfix/<issue-number>-desc` | Critical production bugs | `hotfix/99-fix-auth-bypass` |
| `fix/<issue-number>-desc` | Non-urgent bug fixes | `fix/55-goal-queue-json-corruption` |
| `release/vX.Y.Z` | Release preparation branches | `release/v0.3.0` |
| `docs/<topic>` | Documentation-only changes | `docs/update-api-reference` |
| `chore/<topic>` | Maintenance (deps, CI, config) | `chore/bump-fastapi-0-136` |

Rules:
- Always branch from `main` (never from another feature branch).
- Use kebab-case, all lowercase.
- Include the GitHub issue number when one exists.

```bash
git checkout main
git pull origin main
git checkout -b feature/42-redis-goal-queue
```

---

## Commit Message Format

Write commit messages as a single **imperative sentence-case** summary line, optionally followed by a blank line and a body.

```
Add health endpoint to FastAPI server

Registers GET /health on the FastAPI app instance. Returns a JSON
payload from `build_health_payload()` with provider status and the
run_enabled flag. No auth required so readiness probes work without
a token.

Closes #31
```

### Rules

- **Sentence-case**: capitalise the first word; no trailing period.
- **Imperative mood**: "Add", "Fix", "Remove", "Update" — not "Added", "Fixes", "Removing".
- **Subject line ≤ 72 characters.**
- **Reference issues** in the body with `Closes #N` or `Refs #N`.
- Do **not** use Conventional Commits prefixes (`feat:`, `fix:`) — they conflict with AURA's changelog automation which parses the sentence directly.

---

## Running Tests

```bash
# Run the full test suite (AURA_SKIP_CHDIR=1 prevents chdir to a project root)
AURA_SKIP_CHDIR=1 python3 -m pytest

# Run a specific test file
AURA_SKIP_CHDIR=1 python3 -m pytest tests/test_goal_queue.py -v

# Run with coverage
AURA_SKIP_CHDIR=1 python3 -m pytest --cov=aura_cli --cov=core --cov=agents \
    --cov-report=term-missing

# Run only fast unit tests (exclude slow integration tests)
AURA_SKIP_CHDIR=1 python3 -m pytest -m "not slow"
```

> **Why `AURA_SKIP_CHDIR=1`?**  
> The CLI entrypoint (`aura_cli/entrypoint.py`) changes the working directory to the project root when it detects a project. Setting this env var disables that behaviour so tests run from the repo root and don't pick up a user's real `aura.config.json`.

### Minimum requirements before opening a PR

- All tests pass on Python 3.11: `python3.11 -m pytest`
- No new tests are skipped without a comment explaining why.
- Coverage does not drop below the baseline measured in CI.

---

## Running Lint

```bash
# Check for lint errors
ruff check .

# Check formatting (does not modify files)
ruff format --check .

# Auto-fix lint and format in one pass
ruff check --fix . && ruff format .
```

The ruff configuration lives in `ruff.toml`. Do not disable rules inline (`# noqa`) without a comment explaining why the rule does not apply.

---

## Running Security Checks

```bash
# Run bandit on all first-party source trees (-ll = LOW and above)
bandit -r aura_cli/ core/ agents/ -ll

# For CI (produces machine-readable output)
bandit -r aura_cli/ core/ agents/ -ll -f json -o bandit-report.json
```

### What counts as a blocker

- Any `HIGH` severity finding in **new code** introduced by the PR is a blocker.
- `MEDIUM` findings must be acknowledged with a comment or suppression note.
- `LOW` findings in new code should be fixed where practical.

Do not suppress bandit findings with `# nosec` without a co-author review and a linked issue.

---

## Pull Request Checklist

Before marking your PR as "Ready for Review", confirm each item:

```
□ All acceptance criteria from the linked issue pass
□ New code has unit tests; no coverage drops below baseline
□ pytest green on Python 3.11  (AURA_SKIP_CHDIR=1 python3.11 -m pytest)
□ ruff check . passes with zero errors
□ ruff format --check . passes
□ bandit -r aura_cli/ core/ agents/ -ll returns no HIGH findings in new code
□ Reviewed by ≥ 1 contributor (other than the author)
□ Docstrings on all new public functions and classes
□ CHANGELOG.md updated under [Unreleased] with a human-readable entry
□ No TODOs in new code without a linked GitHub issue (e.g. # TODO(#42): ...)
□ .env.example updated if any new environment variables were added
□ ADR written in docs/adr/ if the PR introduces an architectural decision
```

### CHANGELOG.md format

Add an entry under `## [Unreleased]` in `CHANGELOG.md` following [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions:

```markdown
## [Unreleased]

### Added
- Redis-backed `GoalQueue` when `REDIS_ENABLED=true` (#42)

### Fixed
- Goal queue JSON corruption on concurrent writes (#55)

### Changed
- `aura_cli/server.py` split into `aura_cli/api/` package (ADR-001)
```

---

## EXPERIMENTAL Feature Convention

Some AURA CLI commands are gated behind a feature flag while they are being stabilised. These are marked `[EXPERIMENTAL]`.

### What it means

An `[EXPERIMENTAL]` command or flag:
- May change behaviour or be removed without a deprecation period.
- Is not covered by the backward-compatibility guarantee.
- May have known limitations documented in the help text.
- Requires explicit opt-in (a flag or env var) to activate.

### How to add it to a new command

1. **Add the flag to the CLI command definition** (in `aura_cli/cli_main.py` or the relevant subcommand file):

```python
@app.command()
def my_new_command(
    experimental: bool = typer.Option(
        False,
        "--experimental",
        help="[EXPERIMENTAL] Enable the new orchestrator pipeline. May change without notice.",
        envvar="AURA_ENABLE_MY_NEW_COMMAND",
    ),
    ...
):
    ...
```

2. **Prefix the command's help string with `[EXPERIMENTAL]`:**

```python
@app.command(help="[EXPERIMENTAL] Run goals through the new multi-agent pipeline.")
def pipeline(...):
    ...
```

3. **Emit a warning at runtime** when the experimental path is taken:

```python
from core.logging_utils import log_json
if experimental:
    log_json("WARN", "experimental_feature_enabled", details={"feature": "my_new_command"})
    typer.echo("⚠️  [EXPERIMENTAL] This feature is unstable and may change without notice.", err=True)
```

4. **Document the flag in `.env.example`** under the `# Feature Flags` section.

5. **Open a tracking issue** to graduate or retire the feature, and link it in the help text.

---

## Agent Development

If you are contributing a new agent or modifying an existing one:

- Agent source files live in `agents/`.
- Agent definitions for Copilot CLI live in `.github/agents/`.
- Follow the existing agent format: frontmatter with `name`/`description`, then a markdown body with **Responsibilities**, **Memory Model**, **Interfaces**, and **Failure Modes Guarded Against** sections.
- Add unit tests in `tests/test_agents/`.
- Test your agent locally using the [Copilot CLI](https://gh.io/customagents/cli).

---

## Reporting Issues

- Search [existing issues](https://github.com/aura-cli/aura-cli/issues) before opening a new one.
- Use the **Bug Report** or **Feature Request** issue template.
- For bugs: include the full command run, the error output, your Python version (`python3 --version`), and OS.
- For security vulnerabilities: see `docs/SECURITY.md` — do **not** open a public issue.

---

## Code of Conduct

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By participating you agree to uphold these standards.

---

## Questions?

Open a [Discussion](https://github.com/aura-cli/aura-cli/discussions) or comment in an existing issue thread.

Thank you for helping make AURA CLI better!
