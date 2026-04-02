# AURA CLI

**Autonomous software development platform with multi-agent loop**

AURA (Autonomous Unified Reasoning Agent) is an AI-powered CLI tool that autonomously develops, tests, and deploys software through a multi-agent orchestration system.

## Features

- 🤖 **Multi-Agent Orchestration**: Planner, Coder, Critic, and Debugger agents working together
- 🎯 **Goal-Driven Development**: Describe what you want, AURA figures out how
- 🔄 **Iterative Improvement**: Continuous learning from successes and failures
- 🧠 **Memory System**: Remembers past decisions and patterns
- 🌐 **Web Dashboard**: Monitor and control operations in real-time
- 🔗 **GitHub Integration**: Automatic PR creation and code review
- 🐳 **Cloud Native**: Docker and Kubernetes deployment ready

## Quick Start

### Installation

```bash
pip install aura-cli
```

### Run Your First Goal

```bash
aura goal add "Create a REST API for managing todos"
aura goal run
```

### Start the Web Dashboard

```bash
# Terminal 1: Start API server
aura api run

# Terminal 2: Start Web UI
cd web-ui && npm install && npm run dev
```

Then open http://localhost:3000

## Architecture

AURA uses a closed-loop orchestration system:

1. **INGEST**: Analyze goal and gather context
2. **PLAN**: Generate step-by-step implementation plan
3. **CRITIQUE**: Review plan for issues
4. **SYNTHESIZE**: Merge plan with feedback
5. **ACT**: Generate code changes
6. **VERIFY**: Run tests and validation
7. **REFLECT**: Learn from results

## Documentation

- [Installation Guide](getting-started/installation.md)
- [CLI Reference](user-guide/cli.md)
- [Web Dashboard](user-guide/dashboard.md)
- [Architecture Overview](development/architecture.md)
- [Deployment Guide](deployment/docker.md)

## License

MIT License - see [LICENSE](https://github.com/asshat1981ar/aura-cli/blob/main/LICENSE) for details.
