# AURA CLI

<p align="center">
  <img src="https://raw.githubusercontent.com/asshat1981ar/aura-cli/main/docs/assets/logo.png" alt="AURA CLI Logo" width="200">
</p>

<p align="center">
  <a href="https://github.com/asshat1981ar/aura-cli/actions/workflows/ci.yml">
    <img src="https://github.com/asshat1981ar/aura-cli/actions/workflows/ci.yml/badge.svg" alt="CI Status">
  </a>
  <a href="https://codecov.io/gh/asshat1981ar/aura-cli">
    <img src="https://codecov.io/gh/asshat1981ar/aura-cli/branch/main/graph/badge.svg" alt="Coverage">
  </a>
  <a href="https://pypi.org/project/aura-cli/">
    <img src="https://img.shields.io/pypi/v/aura-cli.svg" alt="PyPI Version">
  </a>
  <a href="https://pypi.org/project/aura-cli/">
    <img src="https://img.shields.io/pypi/pyversions/aura-cli.svg" alt="Python Versions">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  </a>
  <a href="https://github.com/asshat1981ar/aura-cli/stargazers">
    <img src="https://img.shields.io/github/stars/asshat1981ar/aura-cli?style=social" alt="GitHub Stars">
  </a>
</p>

---

**AURA** is an autonomous software development platform that accepts natural-language goals and runs a **10-phase multi-agent pipeline** to design, implement, test, and commit changes with minimal human intervention.

```
┌─────────────────────────────────────────────────────────────────┐
│                   10-Phase Multi-Agent Pipeline                   │
│                                                                   │
│   ingest → plan → critique → code → apply → verify → reflect    │
│     ↑                                              ↓              │
│   archive ← evolve ← adapt ←────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## ✨ Features

- 🤖 **Autonomous Development**: Natural language goals → production code
- 🔁 **10-Phase Pipeline**: Ingest, plan, critique, code, apply, verify, reflect, adapt, evolve, archive
- 🧠 **Multi-Agent System**: Specialized agents for planning, coding, review, and verification
- 🔌 **MCP Integration**: Model Context Protocol for extensible tool support
- 🛡️ **Safety First**: Sandboxed execution, input validation, autonomous apply policies
- 📊 **Observability**: Rich logging, metrics, and WebSocket real-time updates
- 🔐 **Enterprise Security**: JWT authentication, rate limiting, secret management

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Git
- (Optional) Redis for caching

### Installation

```bash
# Install from PyPI (recommended)
pip install aura-cli

# Or install from source
git clone https://github.com/asshat1981ar/aura-cli.git
cd aura-cli
pip install -e ".[dev]"
```

### Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Generate a secure JWT secret
python3 -c "import secrets; print(secrets.token_urlsafe(43))"

# Edit .env and set:
#   AURA_JWT_SECRET=<your-generated-secret>
#   OPENAI_API_KEY=<your-openai-key>  # or ANTHROPIC_API_KEY
```

### Run Your First Goal

```bash
# Run a single goal
aura goal once "Add input validation to the login endpoint"

# Or start the autonomous loop
aura goal run

# Check system health
aura doctor
```

---

## 📖 Command Reference

| Command | Description | Example |
|---------|-------------|---------|
| `aura goal once "<goal>"` | Run a single goal | `aura goal once "Fix typo in README"` |
| `aura goal run` | Run the goal queue | `aura goal run --dry-run` |
| `aura goal add "<goal>"` | Add goal to queue | `aura goal add "Refactor auth" --run` |
| `aura goal status` | Show queue status | `aura goal status --json` |
| `aura doctor` | System health check | `aura doctor --fix` |
| `aura config` | Show configuration | `aura config list` |
| `aura mcp tools` | List MCP tools | `aura mcp tools` |
| `aura memory search` | Search memory | `aura memory search "auth pattern"` |
| `aura agent list` | List agents | `aura agent list` |
| `aura sadd run` | Run SADD workflow | `aura sadd run --spec design.md` |
| `aura innovate start` | Start innovation session | `aura innovate start "How to improve X?"` |

See the [full CLI Reference](docs/CLI_REFERENCE.md) for complete documentation.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         AURA CLI Architecture                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   CLI Layer │───▶│   Typer     │───▶│  Command Handlers   │  │
│  │   (aura)    │    │   Router    │    │  (aura_cli/commands)│  │
│  └─────────────┘    └─────────────┘    └──────────┬──────────┘  │
│                                                   │              │
│  ┌─────────────┐    ┌─────────────┐              ▼              │
│  │   REST API  │───▶│   FastAPI   │───▶│  Core Orchestrator  │  │
│  │  (:8001)    │    │   Server    │    │  (core/orchestrator)│  │
│  └─────────────┘    └─────────────┘    └──────────┬──────────┘  │
│                                                   │              │
│                              ┌────────────────────┼──────────┐  │
│                              ▼                    ▼          │  │
│  ┌─────────────┐    ┌─────────────────┐   ┌──────────────┐  │  │
│  │   Memory    │◀───│  10-Phase Agent │   │   MCP Tools  │  │  │
│  │   System    │    │     Pipeline    │──▶│   Registry   │  │  │
│  └─────────────┘    └─────────────────┘   └──────────────┘  │  │
│                              │                              │  │
│                    ┌─────────┼─────────┐                    │  │
│                    ▼         ▼         ▼                    │  │
│              ┌────────┐ ┌────────┐ ┌────────┐              │  │
│              │Planner │ │ Coder  │ │Verifier│              │  │
│              └────────┘ └────────┘ └────────┘              │  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Path | Description |
|-----------|------|-------------|
| CLI | `aura_cli/cli_main.py` | Typer-based command interface |
| API Server | `aura_cli/server.py` | FastAPI REST + WebSocket endpoints |
| Orchestrator | `core/orchestrator.py` | 10-phase pipeline loop |
| Agents | `agents/` | Specialized pipeline agents |
| Memory | `memory/` | SQLite, JSONL, Redis tiers |
| MCP Tools | `tools/` | Model Context Protocol servers |

---

## ⚙️ Configuration

### Configuration Precedence (highest → lowest)

| Priority | Source |
|----------|--------|
| 1 | Environment variables / `.env` |
| 2 | `aura.config.json` (or `AURA_CONFIG_PATH`) |
| 3 | `settings.json` (model routing, provider config) |
| 4 | Built-in defaults |

### Key Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AURA_JWT_SECRET` | Yes | — | JWT signing key (43+ chars) |
| `OPENAI_API_KEY` | Yes* | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | Yes* | — | Anthropic API key (alternative) |
| `AURA_ENV` | No | `development` | Runtime environment |
| `AURA_LOG_LEVEL` | No | `info` | Log verbosity |
| `AURA_API_HOST` | No | `0.0.0.0` | Server bind address |
| `AURA_API_PORT` | No | `8001` | Server bind port |
| `REDIS_URL` | No | — | Redis connection string |
| `AURA_DRY_RUN` | No | `false` | Simulate without changes |

*At least one LLM provider key is required.

### Configuration Files

| File | Purpose |
|------|---------|
| `.env` | Runtime secrets and overrides (never commit) |
| `aura.config.json` | Project configuration |
| `settings.json` | Model routing and provider selection |
| `.mcp.json` | Repo-local MCP server registry |
| `aura_auth.db` | JWT revocation store (never commit) |

---

## 🧪 Testing

```bash
# Fast regression suite (~4-6 seconds)
python3 -m pytest tests/test_auth.py tests/test_sanitizer.py \
  tests/test_server_api.py tests/test_cli_exit_codes.py \
  tests/test_correlation.py tests/test_config_schema.py \
  -v --timeout=30

# With coverage
python3 -m pytest tests/ -v --timeout=30 \
  --cov=aura_cli --cov=core --cov=agents \
  --cov-report=term-missing

# All tests with triage (handles hanging tests)
python3 scripts/triage_tests.py --timeout 30
```

Required test environment:
```bash
export AURA_SKIP_CHDIR=1
export AURA_TEST_MODE=1
```

---

## 🐳 Docker

```bash
# Development stack (includes n8n, observability)
docker compose up

# Production stack
docker compose -f docker-compose.prod.yml up

# Build production image
docker build -t aura-cli:latest .

# Run with custom config
docker run -v $(pwd)/aura.config.json:/app/aura.config.json aura-cli:latest
```

---

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `AURA_JWT_SECRET not set` | Generate with `python3 -c "import secrets; print(secrets.token_urlsafe(43))"` |
| `ModuleNotFoundError` | Run `pip install -e ".[dev]"` from repo root |
| `Port 8001 already in use` | Set `AURA_API_PORT=8002` or kill existing process |
| `Redis connection failed` | Start Redis or set `REDIS_ENABLED=false` |
| `Tests hang indefinitely` | Always use `--timeout=30` flag with pytest |
| `Permission denied` | Ensure `AURA_SKIP_CHDIR=1` is set for tests |

### Debug Mode

```bash
# Enable debug logging
export AURA_LOG_LEVEL=debug

# Run with verbose output
aura goal once "Test goal" --verbose

# Check system health
aura doctor --json

# View recent logs
aura logs --tail 100
```

### Getting Help

```bash
# Show help for any command
aura --help
aura goal --help
aura goal once --help

# Generate diagnostic report
aura diag

# Check MCP server status
aura mcp status
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Start for Contributors

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/aura-cli.git
cd aura-cli

# 2. Set up development environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. Install pre-commit hooks
pre-commit install

# 4. Run tests
python3 -m pytest tests/ -v --timeout=30

# 5. Create a branch and make changes
git checkout -b feature/my-feature

# 6. Submit a pull request
```

### Development Commands

```bash
# Lint
python3 -m ruff check .

# Type check
python3 -m mypy aura_cli core agents

# Security scan
python3 -m bandit -r aura_cli core agents -ll

# Regenerate CLI docs
python3 scripts/generate_cli_reference.py

# Validate config
python3 scripts/validate_config.py
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [CLI Reference](docs/CLI_REFERENCE.md) | Complete command reference |
| [Command Docs](docs/commands/) | Per-command detailed guides |
| [API Guide](docs/API_GUIDE.md) | REST API documentation |
| [User Guide](docs/USER_GUIDE.md) | End-user guide |
| [Architecture](docs/INTEGRATION_MAP.md) | System architecture |
| [ADRs](docs/adr/INDEX.md) | Architecture Decision Records |
| [Memory Architecture](docs/MEMORY_ARCHITECTURE.md) | Memory tier system |
| [MCP Servers](docs/MCP_SERVERS.md) | MCP integration |
| [Security](docs/SECURITY.md) | Security policy |
| [Threat Model](docs/THREAT_MODEL.md) | Threat model analysis |
| [Contributing](CONTRIBUTING.md) | Contribution guidelines |
| [Changelog](CHANGELOG.md) | Version history |

### 📖 Documentation Site

View the full documentation at [https://asshat1981ar.github.io/aura-cli](https://asshat1981ar.github.io/aura-cli) or run locally:

```bash
# Install mkdocs
pip install mkdocs-material mkdocstrings

# Serve locally
mkdocs serve

# Build site
mkdocs build
```

---

## 🛡️ Security

AURA implements multiple security layers:

- **JWT Authentication**: All API endpoints require valid Bearer tokens
- **Rate Limiting**: Token-bucket algorithm per endpoint/user
- **Sandbox Isolation**: Untrusted code runs in subprocess with tempdir
- **Input Sanitization**: All user input passes through `core/sanitizer.py`
- **Autonomous Apply Policy**: Stale-snippet overwrites blocked by default
- **Secret Management**: Secure credential storage with keyring

See [SECURITY.md](docs/SECURITY.md) and [THREAT_MODEL.md](docs/THREAT_MODEL.md) for details.

---

## 📈 Performance

Benchmarks on reference hardware (AMD Ryzen 9, 32GB RAM):

| Metric | Value |
|--------|-------|
| CLI startup time | ~150ms |
| Goal ingestion | ~500ms |
| Pipeline cycle | ~5-30s (depends on LLM) |
| API request latency (p99) | <100ms |
| Memory footprint | ~150MB base |

See [Performance Guide](docs/performance.md) for optimization strategies.

---

## 🗺️ Roadmap

See [ROADMAP_PRD_SERIES.md](docs/ROADMAP_PRD_SERIES.md) for planned features and development timeline.

Highlights:
- Enhanced multi-agent coordination
- Additional LLM provider support
- Improved sandbox capabilities
- Web UI enhancements
- Enterprise SSO integration

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 💬 Community

- [GitHub Issues](https://github.com/asshat1981ar/aura-cli/issues) — Bug reports and feature requests
- [GitHub Discussions](https://github.com/asshat1981ar/aura-cli/discussions) — Q&A and general discussion
- [Discord](https://discord.gg/aura-cli) — Community chat (coming soon)

---

<p align="center">
  Built with ❤️ by the AURA team and contributors
</p>
