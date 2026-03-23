# Contributing to aura-cli

Thank you for your interest in contributing to aura-cli! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- A GitHub account

### Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/aura-cli.git
   cd aura-cli
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate     # Windows
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Branch Strategy

- `main` — stable, production-ready code
- `feature/*` — new features (branch from `main`)
- `fix/*` — bug fixes (branch from `main`)
- `docs/*` — documentation changes

Always create a new branch for your work:
```bash
git checkout -b feature/your-feature-name
```

## Making Changes

### Code Style

- We use **Ruff** for linting and formatting (configured in `ruff.toml`)
- Run `ruff check .` before committing
- Run `ruff format .` to auto-format
- Pre-commit hooks will catch most style issues automatically

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `refactor:` — code restructuring without behavior change
- `test:` — adding or updating tests
- `chore:` — maintenance tasks

Example: `feat: add retry logic to agent orchestrator`

### Testing

- Run tests with: `python -m pytest tests/`
- Add tests for any new functionality
- Ensure all existing tests pass before submitting a PR

## Submitting a Pull Request

1. Push your branch to your fork
2. Open a Pull Request against `main` on the upstream repository
3. Fill out the PR template completely
4. Ensure CI checks pass
5. Request a review

### PR Guidelines

- Keep PRs focused — one feature or fix per PR
- Include a clear description of what changed and why
- Link any related issues (e.g., "Closes #42")
- Add tests for new functionality
- Update documentation if behavior changes

## Reporting Issues

- Use the issue templates provided (Bug Report or Feature Request)
- Search existing issues before creating a new one
- Include reproduction steps for bugs
- Be specific about expected vs. actual behavior

## Agent Development

If you are contributing a new agent or modifying an existing one:

- Agent definitions live in `.github/agents/`
- Follow the existing agent format (frontmatter with name/description, then markdown body)
- Include: Responsibilities, Memory Model, Interfaces, and Failure Modes Guarded Against
- Test your agent locally using the [Copilot CLI](https://gh.io/customagents/cli)

## Code of Conduct

This project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold these standards.

## Questions?

If you have questions about contributing, open a Discussion or reach out in an existing issue thread.

Thank you for helping make aura-cli better!
