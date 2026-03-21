# AURA CLI Repository - Improvement Recommendations

This document provides comprehensive suggestions for improving code quality, testing, documentation, and overall repository health based on analysis of recent PRs and issues.

## 🧪 Testing Improvements

### 1. Add Integration Test Suite (Issue #207)
**Priority:** High
**Why:** PR #206 adds 7 major modules with only unit tests. Integration tests are critical for validating end-to-end behavior.

**Recommended Test Structure:**
```python
# tests/integration/test_nbest_integration.py
def test_nbest_full_cycle_with_critic_tournament():
    """Test N-best generation with real model, sandbox, and critic."""
    # Setup
    config = {"n_best_candidates": 3}
    orchestrator = create_test_orchestrator(config)

    # Execute
    result = orchestrator.run_cycle("Add a simple function", dry_run=True)

    # Verify
    assert "nbest" in result.get("metadata", {})
    assert result["metadata"]["nbest"]["winner_variant"] in [0, 1, 2]
    assert result["metadata"]["nbest"]["candidates_scored"] == 3

# tests/integration/test_hooks_integration.py
def test_hook_blocking_prevents_phase_execution():
    """Test that hook with exit 2 blocks phase execution."""
    # Create temp hook script
    hook_script = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh')
    hook_script.write('#!/bin/bash\nexit 2\n')
    hook_script.close()
    os.chmod(hook_script.name, 0o755)

    config = {
        "hooks": {
            "pre_apply": {
                "command": hook_script.name,
                "on_failure": "block"
            }
        }
    }

    orchestrator = create_test_orchestrator(config)
    result = orchestrator.run_cycle("Add test", dry_run=True)

    # Verify apply phase was blocked
    assert "_blocked_by_hook" in result.get("phase_outputs", {}).get("apply", {})
```

### 2. Increase Test Coverage for Critical Paths
**Files needing more coverage:**
- `core/hooks.py` - Hook execution, environment passing, output parsing
- `core/mcp_events.py` - SSE streaming, event bus pub/sub
- `core/a2a/*.py` - Agent discovery, task delegation
- `memory/consolidation.py` - Memory pruning, deduplication

**Coverage Target:** 80%+ for all new modules

### 3. Add Property-Based Tests
Use `hypothesis` to test invariants:
```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=0.1, max_value=1.0), min_size=1, max_size=10))
def test_nbest_temperatures_always_produce_candidates(temps):
    """Property: Any valid temperature list should produce valid candidates."""
    engine = NBestEngine(n_candidates=len(temps), temperature_spread=temps)
    # Test that candidates are generated without crashes
```

### 4. Add Performance Regression Tests
Track key metrics over time:
```python
def test_orchestrator_cycle_time_regression():
    """Ensure cycle time doesn't regress beyond acceptable bounds."""
    start = time.time()
    orchestrator.run_cycle("Simple task", dry_run=True)
    duration = time.time() - start

    # Alert if cycle time exceeds threshold (see Issue #177)
    assert duration < 500, f"Cycle time {duration}ms exceeds 500ms threshold"
```

## 📝 Documentation Improvements

### 1. API Documentation
**Missing:** Comprehensive API docs for new modules

**Recommended:**
```bash
# Add Sphinx or MkDocs setup
pip install sphinx sphinx-rtd-theme

# Generate docs
sphinx-quickstart docs/
# Configure autodoc to pull from docstrings
# Build to docs/build/html
```

### 2. Architecture Decision Records (ADRs)
Document major design decisions:
```markdown
# ADR-001: N-Best Code Generation with Critic Tournament

## Context
Single-path code generation can miss better solutions due to model sampling variance.

## Decision
Implement N-best generation with temperature variation and critic-based selection.

## Consequences
+ Higher quality code through diversity
+ Better handling of complex tasks
- Increased latency (3x model calls)
- Higher API costs

## Status
Accepted (PR #206)
```

### 3. Update README with New Features
Add sections for:
- A2A protocol usage and discovery
- Hook system configuration
- Confidence-based routing
- Experiment tracking
- Memory consolidation

### 4. Add Troubleshooting Guide
Common issues and solutions:
```markdown
# Troubleshooting Guide

## Pre-commit hooks failing
**Error:** `detect-secrets: FileNotFoundError: .secrets.baseline`
**Solution:** Run `detect-secrets scan > .secrets.baseline` or remove `--baseline` arg

## A2A discovery fails
**Error:** `Connection refused on port 8000`
**Solution:** Check PORT env var matches AgentCard configuration

## Memory consolidation not running
**Check:** Cycle counter is incrementing and config toggle is enabled
```

## 🔒 Security Enhancements

### 1. Add Security Policy
Create `SECURITY.md`:
```markdown
# Security Policy

## Reporting a Vulnerability
Email security@... or open a private security advisory.

## Security Considerations
- Never commit API keys or tokens
- Use AGENT_API_TOKEN for production deployments
- Review hook commands carefully (shell injection risk)
- A2A endpoints should require authentication in production
```

### 2. Add SAST to CI
```yaml
# .github/workflows/security.yml
name: Security Scan
on: [push, pull_request]
jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Bandit
        run: |
          pip install bandit
          bandit -r . -f json -o bandit-report.json
      - name: Upload results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: bandit-report.json
```

### 3. Add Dependency Scanning
```yaml
# Enable Dependabot
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

## 🏗️ Code Quality Improvements

### 1. Add Pre-commit Hooks for Type Checking
```yaml
# .pre-commit-config.yaml
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--strict, --ignore-missing-imports]
```

### 2. Add Complexity Monitoring
```bash
# Track cyclomatic complexity
pip install radon
radon cc -a core/ agents/

# Add to CI
radon cc --min B core/ agents/ || exit 1
```

### 3. Enforce Docstring Coverage
```python
# Add to CI
interrogate -v --fail-under 80 core/ agents/ aura_cli/
```

### 4. Add Code Review Checklist Template
```markdown
# .github/pull_request_template.md
## Checklist
- [ ] Added tests for new functionality
- [ ] Updated documentation
- [ ] Ran full test suite locally
- [ ] No security vulnerabilities introduced
- [ ] Backwards compatible or migration path documented
- [ ] Performance impact assessed
- [ ] Added logging for error paths
- [ ] Type hints added
```

## 🚀 CI/CD Improvements

### 1. Add Multi-Python Version Testing
```yaml
# .github/workflows/ci.yml
strategy:
  matrix:
    python-version: ["3.10", "3.11", "3.12"]
    os: [ubuntu-latest, macos-latest, windows-latest]
```

### 2. Add Benchmark Tracking
```yaml
# Track performance over time
- name: Run Benchmarks
  run: python scripts/benchmark_loop.py --json > benchmarks.json

- name: Store Benchmark Results
  uses: benchmark-action/github-action-benchmark@v1
  with:
    tool: 'customBiggerIsBetter'
    output-file-path: benchmarks.json
```

### 3. Add Automated Changelog
```yaml
- name: Generate Changelog
  uses: github-changelog-generator/github-changelog-generator-action@v1
  with:
    token: ${{ secrets.GITHUB_TOKEN }}
```

### 4. Add Release Automation
```yaml
# .github/workflows/release.yml (already exists, ensure it's comprehensive)
# Add:
- Create GitHub Release with notes
- Upload wheel and sdist to PyPI
- Create Docker image and push to registry
- Update documentation site
- Post announcement to discussions
```

## 📦 Packaging Improvements

### 1. Add Optional Dependencies
```toml
# pyproject.toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "hypothesis",
    "mypy",
    "ruff",
]
a2a = [
    "httpx>=0.24.0",
]
mcp = [
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
]
all = [
    "aura-cli[dev,a2a,mcp]",
]
```

### 2. Add Entry Points
```toml
[project.scripts]
aura = "aura_cli.cli_main:main"
aura-server = "aura_cli.server:main"
```

### 3. Add Dockerfile.dev
```dockerfile
# Dockerfile.dev - development image with all tools
FROM python:3.12-slim

WORKDIR /app

# Install dev dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements.txt -r requirements-dev.txt

# Mount source as volume for live reload
VOLUME /app

CMD ["python", "-m", "pytest", "--watch"]
```

## 🎯 Feature Enhancements

### 1. Add Observability (Issue #209 related)
```python
# core/telemetry.py
from opentelemetry import trace, metrics

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

cycle_counter = meter.create_counter("aura.cycles.total")
cycle_duration = meter.create_histogram("aura.cycle.duration")

@tracer.start_as_current_span("orchestrator.run_cycle")
def run_cycle(goal, dry_run):
    with cycle_duration.record_context():
        # ... cycle logic
        cycle_counter.add(1, {"success": True})
```

### 2. Add Structured Configuration Validation
```python
# Use pydantic for config
from pydantic import BaseModel, Field, validator

class AuraConfig(BaseModel):
    n_best_candidates: int = Field(default=1, ge=1, le=10)
    api_key: str = Field(..., min_length=20)
    memory_consolidation: MemoryConsolidationConfig = MemoryConsolidationConfig()

    @validator('api_key')
    def api_key_not_placeholder(cls, v):
        if 'YOUR_KEY' in v or 'placeholder' in v.lower():
            raise ValueError('API key appears to be a placeholder')
        return v

# Load and validate
config = AuraConfig.parse_file("aura.config.json")
```

### 3. Add Plugin System
```python
# core/plugins.py
class PluginInterface:
    """Base class for AURA plugins."""
    def on_cycle_start(self, goal: str) -> None:
        pass

    def on_cycle_end(self, result: dict) -> None:
        pass

class PluginManager:
    def __init__(self):
        self.plugins: list[PluginInterface] = []

    def load_from_config(self, config: dict):
        for plugin_spec in config.get("plugins", []):
            module = importlib.import_module(plugin_spec["module"])
            plugin_class = getattr(module, plugin_spec["class"])
            self.plugins.append(plugin_class(**plugin_spec.get("config", {})))
```

### 4. Add Skill Correlation Learning (Issue #210)
```python
# core/skill_correlation.py
class SkillCorrelationMatrix:
    """Track which skills succeed together."""
    def __init__(self):
        self.matrix: dict[tuple[str, str], float] = {}
        self.success_counts: dict[tuple[str, str], int] = {}

    def record_co_success(self, skill_a: str, skill_b: str):
        """Record that two skills both succeeded in same cycle."""
        key = tuple(sorted([skill_a, skill_b]))
        self.success_counts[key] = self.success_counts.get(key, 0) + 1
        # Update correlation strength
        self.matrix[key] = min(1.0, self.success_counts[key] / 10.0)

    def get_suggested_pairs(self, skill: str, threshold: float = 0.5) -> list[str]:
        """Get skills that correlate well with given skill."""
        suggestions = []
        for (s1, s2), correlation in self.matrix.items():
            if correlation >= threshold:
                if s1 == skill:
                    suggestions.append(s2)
                elif s2 == skill:
                    suggestions.append(s1)
        return suggestions
```

## 🔧 Developer Experience Improvements

### 1. Add Development Setup Script
```bash
#!/bin/bash
# scripts/dev_setup.sh

set -e

echo "Setting up AURA development environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup pre-commit
pre-commit install

# Generate secrets baseline
detect-secrets scan > .secrets.baseline

# Create local config
if [ ! -f aura.config.json ]; then
    cp aura.config.example.json aura.config.json
    echo "Created aura.config.json - please add your API keys"
fi

# Initialize memory directory
mkdir -p memory
touch memory/.gitkeep

echo "✓ Development environment ready!"
echo "Next steps:"
echo "  1. Add your API keys to aura.config.json"
echo "  2. Run: python3 main.py doctor"
echo "  3. Run tests: pytest"
```

### 2. Add VS Code Configuration
```json
// .vscode/settings.json.example
{
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "memory/*.db": true,
    "memory/*.json": true
  }
}
```

### 3. Add Debug Configurations
```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "AURA: Run Single Goal",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/main.py",
      "args": ["--goal", "Add a simple test"],
      "console": "integratedTerminal",
      "env": {
        "AURA_SKIP_CHDIR": "1",
        "AURA_DRY_RUN": "1"
      }
    },
    {
      "name": "AURA: Run Server",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": ["aura_cli.server:app", "--reload"],
      "console": "integratedTerminal"
    }
  ]
}
```

## 📊 Monitoring & Metrics

### 1. Add Health Dashboard
```python
# aura_cli/dashboard.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Simple health dashboard."""
    metrics = {
        "cycles_completed": orchestrator._cycle_count,
        "goals_in_queue": len(runtime["goal_queue"]),
        "memory_size_mb": get_memory_size(),
        "skill_success_rate": calculate_skill_success_rate(),
        "avg_cycle_time_sec": calculate_avg_cycle_time(),
    }

    return f"""
    <html>
    <head><title>AURA Dashboard</title></head>
    <body>
        <h1>AURA Status</h1>
        <ul>
            {''.join(f'<li>{k}: {v}</li>' for k, v in metrics.items())}
        </ul>
    </body>
    </html>
    """
```

### 2. Add Prometheus Metrics Endpoint
```python
from prometheus_client import Counter, Histogram, generate_latest

cycles_total = Counter('aura_cycles_total', 'Total cycles executed')
cycle_duration = Histogram('aura_cycle_duration_seconds', 'Cycle duration')

@app.get("/metrics")
async def prometheus_metrics():
    return Response(generate_latest(), media_type="text/plain")
```

## 🎓 Learning & Documentation

### 1. Add Example Projects
```
examples/
├── simple_webapp/
│   ├── README.md
│   ├── goal.txt
│   └── expected_output/
├── data_analysis/
│   ├── README.md
│   ├── dataset.csv
│   └── goal.txt
└── cli_tool/
    ├── README.md
    └── goal.txt
```

### 2. Add Video Tutorials
- Getting started
- Configuring hooks
- Using A2A protocol
- Debugging failed cycles
- Custom skill development

### 3. Add FAQ
Common questions from users/contributors

### 4. Add Contributing Guide
```markdown
# CONTRIBUTING.md

## Development Workflow
1. Fork the repo
2. Create feature branch
3. Make changes with tests
4. Run full test suite
5. Update documentation
6. Submit PR with detailed description

## Code Style
- Follow PEP 8
- Use type hints
- Write docstrings
- Keep functions small (<50 lines)
- Maximum line length: 100

## Commit Messages
- Use conventional commits format
- Examples: `feat: add A2A discovery`, `fix: memory leak in EventBus`
```

## 🔮 Future Enhancements

### 1. JSON-RPC/stdio Transport (Issue #209)
Defer to post-v0.1.0 but plan architecture now.

### 2. Web UI
Build React/Vue dashboard for monitoring and control.

### 3. Multi-Agent Collaboration
Enable multiple AURA instances to work on sub-tasks in parallel.

### 4. Cloud Deployment Guide
Add docs for deploying to AWS, GCP, Azure.

### 5. Model Fine-Tuning Pipeline
Collect successful cycles to fine-tune domain-specific models.

## ✅ Summary Priority Matrix

| Priority | Category | Items | Timeline |
|----------|----------|-------|----------|
| P0 | Security | Auth fixes, shell injection | Before PR #206 merge |
| P1 | Correctness | Sandbox adapter, git revert, consolidation | Before PR #206 merge |
| P2 | Testing | Integration tests, coverage | With PR #206 or immediately after |
| P3 | Docs | README updates, ADRs | Post v0.1.0 release |
| P4 | DX | Dev scripts, VS Code config | Post v0.1.0 release |
| P5 | Features | Skill correlation, plugins | v0.2.0 cycle |

## 🎯 Immediate Action Items

1. Address all P0/P1 issues in `PR_206_CRITICAL_FIXES.md`
2. Generate `.secrets.baseline` or fix pre-commit config
3. Add integration test framework
4. Update documentation for new features
5. Add security policy
6. Set up dependency scanning
7. Create development setup script
8. Add contributing guide
9. Plan v0.1.1 bug fix release if needed
10. Plan v0.2.0 feature roadmap
