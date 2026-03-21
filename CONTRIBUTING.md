# Contributing to AURA CLI

Thank you for your interest in contributing to AURA CLI! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing Guidelines](#testing-guidelines)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

Be respectful, constructive, and professional in all interactions. We're building a tool to help developers - let's support each other in that mission.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment (recommended)

### Initial Setup

We provide an automated setup script:

```bash
# Clone the repository
git clone https://github.com/asshat1981ar/aura-cli.git
cd aura-cli

# Run the setup script
./scripts/dev_setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pre-commit
pre-commit install
```

### Configuration

1. Copy the example config:
   ```bash
   cp aura.config.example.json aura.config.json
   ```

2. Add your API keys to `aura.config.json`

3. Create a `.env` file (optional):
   ```bash
   echo "AURA_SKIP_CHDIR=1" > .env
   echo "AURA_API_KEY=your_key_here" >> .env
   ```

### Verify Setup

```bash
# Run health check
python3 main.py doctor

# Run tests
python3 -m pytest

# Try a simple goal
python3 main.py --goal "Say hello" --dry-run
```

## Development Workflow

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/aura-cli.git
cd aura-cli

# Add upstream remote
git remote add upstream https://github.com/asshat1981ar/aura-cli.git
```

### 2. Create a Feature Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 3. Make Changes

- Write code following our [style guidelines](#code-style)
- Add tests for new functionality
- Update documentation as needed
- Run tests locally

### 4. Commit Changes

Follow our [commit message guidelines](#commit-messages):

```bash
git add .
git commit -m "feat: add new feature"
```

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a PR on GitHub.

## Code Style

### Python Style

- **Indentation:** 4 spaces (no tabs)
- **Line length:** Flexible (generous 300 char limit), but prefer <100 for readability
- **Naming conventions:**
  - Functions/variables/modules: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
  - Private members: leading underscore `_private_method`

### Type Hints

Add type hints to all new code:

```python
def process_goal(goal: str, max_cycles: int = 5) -> dict:
    """Process a goal and return results."""
    ...
```

Use `from __future__ import annotations` for forward references.

### Docstrings

Use Google-style docstrings:

```python
def run_cycle(goal: str, dry_run: bool = False) -> dict:
    """Run a single autonomous development cycle.

    Args:
        goal: Natural language description of what to accomplish
        dry_run: If True, don't write files or make changes

    Returns:
        Dictionary containing cycle results, phase outputs, and metadata

    Raises:
        ValueError: If goal is empty
        RuntimeError: If cycle fails to initialize
    """
    ...
```

### Imports

- Standard library imports first
- Third-party imports second
- Local imports last
- Separate groups with blank line
- Sort alphabetically within groups

```python
import json
import time
from pathlib import Path

import fastapi
import pydantic

from core.logging_utils import log_json
from core.model_adapter import ModelAdapter
```

### Error Handling

- Use specific exceptions over bare `Exception`
- Add context to error messages
- Log errors with structured logging

```python
try:
    result = process_data(input_data)
except ValueError as e:
    log_json("ERROR", "process_failed", details={"error": str(e)})
    raise
```

## Testing Guidelines

### Test Structure

- Use `pytest` framework
- File naming: `test_*.py`
- Class naming: `Test*`
- Method naming: `test_*`

### Test Location

- Unit tests: `tests/test_*.py`
- Integration tests: `tests/integration/test_*.py`
- Core module tests: `core/tests/test_*.py`

### Writing Tests

```python
import pytest
from unittest.mock import MagicMock

def test_feature_with_valid_input():
    """Test feature handles valid input correctly."""
    # Arrange
    input_data = {"key": "value"}

    # Act
    result = process_feature(input_data)

    # Assert
    assert result["status"] == "success"
    assert "data" in result
```

### Running Tests

```bash
# Run all tests
python3 -m pytest

# Run specific test file
python3 -m pytest tests/test_file_tools.py

# Run specific test
python3 -m pytest tests/test_file_tools.py::TestClassName::test_method

# Run with coverage
python3 -m pytest --cov=core --cov=agents --cov-report=html

# Run only fast tests (skip integration)
python3 -m pytest -m "not integration"
```

### Test Coverage

- Aim for 80%+ coverage on new code
- All new features must have tests
- Bug fixes should include regression tests

## Commit Messages

### Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks (dependencies, config, etc.)
- `perf`: Performance improvements

### Examples

```bash
# Simple feature
git commit -m "feat: add N-best code generation"

# Bug fix with scope
git commit -m "fix(orchestrator): correct memory consolidation cycle count"

# With body
git commit -m "feat(hooks): add lifecycle hooks system

Add pre/post phase hooks with block/modify/observe actions.
Hooks run via subprocess with timeout and environment context."

# Breaking change
git commit -m "feat!: change API authentication to bearer tokens

BREAKING CHANGE: API now requires AGENT_API_TOKEN env var"
```

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation (README, CLAUDE.md, etc.)
- [ ] Commit messages follow conventions
- [ ] No merge conflicts with main branch
- [ ] Pre-commit hooks pass

### PR Title

Use the same format as commit messages:

```
feat(agents): add skill correlation learning
```

### PR Description

Include:

1. **Summary**: What does this PR do?
2. **Motivation**: Why is this change needed?
3. **Changes**: Bullet list of key changes
4. **Testing**: How was this tested?
5. **Screenshots**: If applicable (UI changes)
6. **Breaking Changes**: If any
7. **Related Issues**: Link to issues

Example:

```markdown
## Summary
Add N-best code generation with critic tournament selection

## Motivation
Single-path generation misses better solutions. Generate N variants and let critic pick best.

## Changes
- Add `core/nbest.py` with `NBestEngine`
- Integrate into orchestrator Act phase
- Add temperature variation for diversity
- Implement critic scoring system

## Testing
- Unit tests in `tests/test_nbest.py`
- Integration test with real cycle
- Verified with 3 test goals

## Breaking Changes
None (opt-in via `n_best_candidates` config)

## Related Issues
Closes #123
```

### Review Process

1. Automated checks must pass (CI, linting, tests)
2. At least one maintainer approval required
3. Address review feedback promptly
4. Keep PR scope focused (one feature/fix per PR)

### After Merge

- Delete your feature branch
- Update local main branch
- Close related issues if not auto-closed

## Reporting Issues

### Bug Reports

Include:

- **Description**: Clear description of the bug
- **Steps to Reproduce**: Minimal steps to reproduce
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: OS, Python version, AURA version
- **Logs**: Relevant log output (mask API keys!)
- **Additional Context**: Any other relevant information

### Feature Requests

Include:

- **Problem**: What problem does this solve?
- **Proposed Solution**: Your suggested approach
- **Alternatives**: Other approaches considered
- **Examples**: Similar features in other tools
- **Use Case**: When would this be used?

### Security Issues

**Do not open public issues for security vulnerabilities!**

Instead:
1. Email security contact (if provided in SECURITY.md)
2. Or open a [private security advisory](https://docs.github.com/en/code-security/security-advisories)

## Questions?

- Check the [README](README.md)
- Read the [CLAUDE.md](CLAUDE.md) architecture guide
- Review existing issues and PRs
- Ask in discussions

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to AURA CLI! 🚀
