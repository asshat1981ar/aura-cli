# AURA Plugin Development Guide

This guide explains how to build, test, and distribute custom AURA plugins.

---

## Quick Start

```python
# my_plugin/notifier.py
from core.plugin_api import SkillPlugin, on_event, AuraHooks

class SlackNotifier(SkillPlugin):
    name = "slack_notifier"
    description = "Posts cycle results to a Slack channel."

    def run(self, input_data: dict) -> dict:
        webhook = input_data.get("webhook_url")
        message = input_data.get("message", "AURA cycle complete.")
        # ... send HTTP POST to Slack ...
        return {"status": "sent", "message": message}


@on_event(AuraHooks.POST_VERIFY)
async def notify_on_verify(result=None, **kwargs):
    """Called automatically after every verify phase."""
    print(f"[SlackNotifier] verify result: {result}")
```

Register it in `pyproject.toml`:

```toml
[project.entry-points."aura.skills"]
slack_notifier = "my_plugin.notifier:SlackNotifier"
```

Load all installed plugins at runtime:

```python
from core.plugin_api import discover_plugins
discover_plugins()  # loads all `aura.skills` entry-points
```

---

## Plugin API Reference

All stable symbols live in `core/plugin_api.py`.

| Symbol | Type | Purpose |
|--------|------|---------|
| `PLUGIN_API_VERSION` | `str` | Current API version (`"1.0"`) |
| `SkillPlugin` | base class | Inherit to create a skill |
| `AuraHooks` | class | Named event constants |
| `EventBus` | class | Subscribe / publish events |
| `aura_plugin(event)` | decorator | Register a coroutine to an event |
| `on_event(event)` | decorator | Alias for `aura_plugin` |
| `register_skill(name, cls)` | function | Manually register a skill class |
| `get_registered_skills()` | function | Returns `{name: cls}` registry |
| `discover_plugins(group)` | function | Load entry-point plugins |

---

## Available Events

| Constant | Fired when |
|----------|-----------|
| `AuraHooks.ON_CYCLE_START` | A new orchestrator cycle begins |
| `AuraHooks.PRE_APPLY_CHANGES` | Before file changes are written to disk |
| `AuraHooks.POST_VERIFY` | After the verify phase completes |
| `AuraHooks.ON_AGENT_ERROR` | An agent raises an unexpected exception |

Subscribe with the `@on_event` decorator or `EventBus.subscribe()` directly.

---

## Writing a Skill Plugin

1. Subclass `SkillPlugin`.
2. Set `name` (unique identifier) and `description`.
3. Implement `run(input_data) -> dict`.
4. **Never raise** — return `{"error": "..."}` on failure.

```python
class MyAnalyzer(SkillPlugin):
    name = "my_analyzer"
    description = "Counts TODO comments."

    def run(self, input_data: dict) -> dict:
        code = input_data.get("code", "")
        count = code.count("TODO")
        return {"todo_count": count, "status": "ok"}
```

---

## Testing Your Plugin

```python
import unittest
from my_plugin.notifier import SlackNotifier

class TestSlackNotifier(unittest.TestCase):
    def test_run_returns_status(self):
        plugin = SlackNotifier()
        result = plugin.run({"message": "hello"})
        self.assertIn("status", result)
```

Run with: `python3 -m pytest tests/`

---

## Distribution

Publish to PyPI as a normal Python package.
AURA discovers plugins via the `aura.skills` entry-point group.

```toml
[project]
name = "aura-slack-notifier"
version = "0.1.0"

[project.entry-points."aura.skills"]
slack_notifier = "my_plugin.notifier:SlackNotifier"
```

Users install with: `pip install aura-slack-notifier`

---

## Versioning

Check `PLUGIN_API_VERSION` at runtime to guard against incompatible APIs:

```python
from core.plugin_api import PLUGIN_API_VERSION
assert PLUGIN_API_VERSION == "1.0", "Incompatible AURA plugin API"
```

---

## Plugin Isolation

- A plugin that raises an unhandled exception is logged and skipped.
- Plugins run in the same process as AURA — use `asyncio` event handlers carefully.
- File system access is unrestricted; follow the principle of least privilege.
