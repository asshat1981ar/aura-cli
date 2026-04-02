# Installation

## Requirements

- Python 3.10 or higher
- Git
- (Optional) Node.js 18+ for Web UI
- (Optional) Docker for containerized deployment

## Install from PyPI

```bash
pip install aura-cli
```

## Install from Source

```bash
git clone https://github.com/asshat1981ar/aura-cli.git
cd aura-cli
pip install -e ".[dev]"
```

## Optional Dependencies

### GitHub Integration
```bash
pip install -e ".[github]"
```

### Web UI Dependencies
```bash
pip install -e ".[webui]"
```

### All Features
```bash
pip install -e ".[dev,github,webui]"
```

## Verify Installation

```bash
aura --help
```

## Configuration

Create a configuration file at `~/.aura/config.json`:

```json
{
  "openai_api_key": "your-api-key",
  "model": "gpt-4",
  "max_cycles": 10,
  "log_level": "INFO"
}
```

Or set environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
export AURA_MODEL="gpt-4"
```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Configuration Reference](configuration.md)
