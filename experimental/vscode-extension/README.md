# AURA CLI for VS Code

Brings the AURA autonomous multi-agent development platform directly into VS Code.

## Requirements
- Python 3.10+
- aura-cli installed at a local path

## Commands
- **AURA: Run Goal** (`Ctrl+Shift+A`) — Send a natural language goal to AURA agents
- **AURA: Stream Goal (Live)** — Stream goal execution with live progress
- **AURA: Agent Status** — Show active agent count

## Configuration
- `aura.projectRoot` — Path to aura-cli project root
- `aura.pythonPath` — Python interpreter (default: `python3`)

## Protocol
Uses JSON-RPC 2.0 over stdio transport (`core/transport.py`). Spawns `python3 main.py transport --root <project>`.
