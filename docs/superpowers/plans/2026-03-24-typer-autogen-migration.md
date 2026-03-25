# Typer + AutoGen Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add AutoGen multi-agent patterns (GroupChat, Swarm, CaptainAgent) as a new brainstorming/ideation subsystem, plus a migration assistant module with 7-step LLM-ready prompts for future Typer CLI migration. Update Dockerfile and CI.

**Architecture:** AutoGen integration adds `agents/autogen_agent.py` that plugs into the existing agent registry and router as a group-chat specialist. The migration assistant module provides structured prompts for a future Typer CLI migration (Typer app creation is a separate follow-up plan). Both share `requirements.txt` and CI updates.

**Tech Stack:** Python 3.10+, typer>=0.12.0, pyautogen>=0.4, rich (existing), pydantic (existing), pytest (existing)

---

## File Structure

### New Files
| File | Responsibility |
|------|----------------|
| `agents/autogen_agent.py` | AutoGen GroupChat agent adapter for the agent registry |
| `aura_cli/migration_assistant.py` | Migration prompts module (7-step LLM-ready templates) |
| `tests/test_deps_available.py` | Verify new dependencies are importable |
| `tests/test_migration_assistant.py` | Tests for migration prompt completeness and safety |
| `tests/test_autogen_agent.py` | Tests for AutoGen agent adapter |
| `tests/test_autogen_registration.py` | Tests for agent registry + router integration |

### Modified Files
| File | Change |
|------|--------|
| `requirements.txt` | Add `typer>=0.12.0`, `pyautogen>=0.4` |
| `pyproject.toml` | Add deps to `[project]` dependencies |
| `agents/registry.py:424` | Add `config=None` kwarg to `default_agents()`, register `autogen_group_chat` |
| `orchestrator_hub/router.py:~60` | Add `"brainstorming"` keywords to `_TASK_KEYWORDS` dict |
| `aura.config.json` | Add `"autogen"` config section |
| `Dockerfile` | Update to multi-stage optimized build with non-root user |
| `.github/workflows/ci.yml` | Add migration assistant test step |

### Unchanged Files (preserve as-is)
- `aura_cli/cli_options.py` — argparse layer kept intact (Typer migration is a future plan)
- `aura_cli/cli_main.py` — command dispatch unchanged
- `core/orchestrator.py` — 10-phase loop unchanged
- `core/git_tools.py`, `core/model_adapter.py` — no changes

---

## Task 1: Update Dependencies

**Files:**
- Create: `tests/test_deps_available.py`
- Modify: `requirements.txt`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write a test that imports the new deps**

```python
# tests/test_deps_available.py
def test_typer_importable():
    import typer
    assert hasattr(typer, "Typer")

def test_autogen_importable():
    import autogen
    assert hasattr(autogen, "AssistantAgent")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_deps_available.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Add dependencies to requirements.txt**

Add these lines to `requirements.txt`:
```
typer>=0.12.0
pyautogen>=0.4
```

Note: The PyPI package is `pyautogen`, not `autogen`. The import name is still `autogen`.

- [ ] **Step 4: Add dependencies to pyproject.toml**

In the `[project]` dependencies list, add:
```toml
"typer>=0.12.0",
"pyautogen>=0.4",
```

- [ ] **Step 5: Install and verify**

Run: `pip install -r requirements.txt && python3 -m pytest tests/test_deps_available.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pyproject.toml tests/test_deps_available.py
git commit -m "feat: add typer and pyautogen dependencies"
```

---

## Task 2: Migration Assistant Module

**Files:**
- Create: `aura_cli/migration_assistant.py`
- Create: `tests/test_migration_assistant.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_migration_assistant.py
import unittest

class TestMigrationPrompts(unittest.TestCase):
    def setUp(self):
        from aura_cli.migration_assistant import MigrationPrompts
        self.prompts = MigrationPrompts

    def test_has_seven_steps(self):
        all_p = self.prompts.get_all_migration_prompts()
        self.assertEqual(len(all_p), 7)

    def test_all_prompts_have_required_fields(self):
        for p in self.prompts.get_all_migration_prompts():
            self.assertIn("step", p)
            self.assertIn("title", p)
            self.assertIn("prompt", p)
            self.assertIsInstance(p["step"], int)
            self.assertGreater(len(p["prompt"]), 100)

    def test_all_prompts_expanded_and_complete(self):
        all_p = self.prompts.get_all_migration_prompts()
        for p in all_p:
            self.assertGreater(len(p["prompt"]), 800)
            self.assertIn("COMPLETE CODE TEMPLATE", p["prompt"])

    def test_no_secrets(self):
        all_text = "\n".join(p["prompt"] for p in self.prompts.get_all_migration_prompts())
        self.assertNotIn("sk-", all_text)
        self.assertNotIn("hardcoded", all_text.lower())

    def test_steps_are_sequential(self):
        all_p = self.prompts.get_all_migration_prompts()
        steps = [p["step"] for p in all_p]
        self.assertEqual(steps, list(range(1, 8)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_migration_assistant.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement MigrationPrompts class**

Create `aura_cli/migration_assistant.py` with the full `MigrationPrompts` class containing all 7 static methods (`step1_typer_cli_migration` through `step7_full_migration_roadmap`) and `get_all_migration_prompts()`. Each step method returns a string containing "COMPLETE CODE TEMPLATE" and detailed migration instructions (800+ chars). Use env-var references (`${ANTHROPIC_API_KEY}`) instead of hardcoded secrets.

The 7 steps are:
1. Typer CLI Migration
2. Two-Agent AutoGen Integration
3. GroupChat for Ideation
4. Swarm Handoff Self-Evolution
5. Hierarchical + Nested Patterns
6. CaptainAgent Auto-Build
7. Full Migration Roadmap

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_migration_assistant.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add aura_cli/migration_assistant.py tests/test_migration_assistant.py
git commit -m "feat: add migration assistant with 7-step LLM-ready prompts"
```

---

## Task 3: AutoGen Group Chat Agent

**Files:**
- Create: `agents/autogen_agent.py`
- Create: `tests/test_autogen_agent.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_autogen_agent.py
import unittest
from unittest.mock import MagicMock, patch

class TestAutoGenGroupChatAgent(unittest.TestCase):
    def test_agent_has_required_attributes(self):
        from agents.autogen_agent import AutoGenGroupChatAgent
        agent = AutoGenGroupChatAgent(brain=MagicMock(), model=MagicMock())
        self.assertEqual(agent.name, "autogen_group_chat")
        self.assertTrue(callable(getattr(agent, "run", None)))

    def test_run_returns_expected_schema(self):
        from agents.autogen_agent import AutoGenGroupChatAgent
        agent = AutoGenGroupChatAgent(brain=MagicMock(), model=MagicMock())
        with patch.object(agent, "_conduct_group_chat", return_value={
            "conversation": "Agent1: idea\nAgent2: refinement",
            "decisions": ["use adapter pattern"],
        }):
            result = agent.run({"goal": "brainstorm auth approach"})
        self.assertIn("conversation", result)
        self.assertIn("decisions", result)
        self.assertIn("participants", result)

    def test_run_stores_to_brain(self):
        from agents.autogen_agent import AutoGenGroupChatAgent
        brain = MagicMock()
        agent = AutoGenGroupChatAgent(brain=brain, model=MagicMock())
        with patch.object(agent, "_conduct_group_chat", return_value={
            "conversation": "test", "decisions": [],
        }):
            agent.run({"goal": "test goal"})
        brain.remember.assert_called_once()

    def test_fallback_when_autogen_unavailable(self):
        from agents.autogen_agent import AutoGenGroupChatAgent
        agent = AutoGenGroupChatAgent(brain=MagicMock(), model=MagicMock())
        # Force fallback path by setting _autogen_available to False
        agent._autogen_available = False
        result = agent.run({"goal": "test"})
        self.assertIn("conversation", result)
        self.assertIn("[fallback]", result["conversation"])
        self.assertEqual(result["decisions"], [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_autogen_agent.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement AutoGenGroupChatAgent**

Create `agents/autogen_agent.py`:

```python
"""AutoGen group-chat agent adapter for AURA's agent registry."""
from __future__ import annotations

from typing import Any, Dict
from core.logging_utils import log_json


class AutoGenGroupChatAgent:
    """Wraps AutoGen GroupChat as an AURA agent with run() interface."""

    name = "autogen_group_chat"

    def __init__(self, brain=None, model=None, config: dict | None = None):
        self.brain = brain
        self.model = model
        self.config = config or {}
        self._autogen_available = self._check_autogen()

    @staticmethod
    def _check_autogen() -> bool:
        try:
            import autogen  # noqa: F401
            return True
        except ImportError:
            return False

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        goal = input_data.get("goal", "")
        if not self._autogen_available:
            log_json("WARN", "autogen_not_available",
                     details={"message": "pyautogen not installed, using fallback"})
            return self._fallback_brainstorm(goal)

        result = self._conduct_group_chat(goal)
        if self.brain:
            self.brain.remember(
                f"AutoGen group chat for: {goal[:100]}. "
                f"Decisions: {result.get('decisions', [])}"
            )
        return {
            "conversation": result.get("conversation", ""),
            "decisions": result.get("decisions", []),
            "participants": self.config.get("agents", []),
        }

    def _conduct_group_chat(self, goal: str) -> Dict[str, Any]:
        from autogen import AssistantAgent, GroupChat, GroupChatManager

        agents_config = self.config.get("agents", [
            {"name": "ideator", "system_message": "Generate creative solutions"},
            {"name": "critic", "system_message": "Find risks and flaws"},
            {"name": "synthesizer", "system_message": "Merge ideas into actionable plan"},
        ])
        llm_config = self.config.get("llm_config", {})
        max_rounds = self.config.get("max_turns", 6)

        agents = [
            AssistantAgent(name=a["name"], system_message=a["system_message"],
                           llm_config=llm_config)
            for a in agents_config
        ]
        groupchat = GroupChat(agents=agents, messages=[], max_round=max_rounds)
        manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        import asyncio
        asyncio.run(manager.a_initiate_chat(agents[0], message=goal))
        messages = [msg.get("content", "") for msg in groupchat.messages]
        return {
            "conversation": "\n".join(messages),
            "decisions": [m for m in messages if "decision:" in m.lower()],
        }

    def _fallback_brainstorm(self, goal: str) -> Dict[str, Any]:
        return {
            "conversation": f"[fallback] Single-agent brainstorm for: {goal}",
            "decisions": [],
            "participants": [],
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_autogen_agent.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add agents/autogen_agent.py tests/test_autogen_agent.py
git commit -m "feat: add AutoGen group-chat agent adapter"
```

---

## Task 4: Register AutoGen Agent in Registry

**Files:**
- Modify: `agents/registry.py:424` — add `config=None` kwarg to `default_agents()` signature
- Modify: `orchestrator_hub/router.py:~60` — add brainstorming keywords to `_TASK_KEYWORDS`
- Modify: `aura.config.json` — add `"autogen"` config section
- Create: `tests/test_autogen_registration.py`

**Important context:** The current `default_agents()` signature is:
```python
def default_agents(brain, model, context_manager=None, skills=None, health_monitor=None):
```
We must add `config=None` as a new kwarg. The `TaskRouter.__init__` requires an `AgentRegistryHub` argument.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_autogen_registration.py
import unittest
from unittest.mock import MagicMock

class TestAutoGenRegistration(unittest.TestCase):
    def test_autogen_agent_in_default_agents(self):
        from agents.registry import default_agents
        agents = default_agents(
            brain=MagicMock(), model=MagicMock(), config={"autogen": {"enabled": True}}
        )
        self.assertIn("autogen_group_chat", agents)

    def test_default_agents_works_without_config(self):
        """Backward compat: config param is optional."""
        from agents.registry import default_agents
        agents = default_agents(brain=MagicMock(), model=MagicMock())
        # Should still work and include autogen with empty config
        self.assertIn("autogen_group_chat", agents)

    def test_router_keywords_include_brainstorming(self):
        from orchestrator_hub.router import _TASK_KEYWORDS
        self.assertIn("brainstorming", _TASK_KEYWORDS)
        self.assertIn("brainstorm", _TASK_KEYWORDS["brainstorming"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_autogen_registration.py -v`
Expected: FAIL (no `config` kwarg, no `brainstorming` in keywords)

- [ ] **Step 3: Add `config` kwarg to `default_agents()` in `agents/registry.py`**

Change the signature at line 424 from:
```python
def default_agents(brain, model, context_manager=None, skills=None, health_monitor=None):
```
To:
```python
def default_agents(brain, model, context_manager=None, skills=None, health_monitor=None, config=None):
```

Then at the end of the function, before `return agent_dict`, add:
```python
    from agents.autogen_agent import AutoGenGroupChatAgent
    agent_dict["autogen_group_chat"] = AutoGenGroupChatAgent(
        brain=brain, model=model, config=(config or {}).get("autogen", {})
    )
```

- [ ] **Step 4: Add brainstorming keywords to `orchestrator_hub/router.py`**

In the `_TASK_KEYWORDS` dict (around line 60), add this entry:

```python
"brainstorming": ["brainstorm", "ideate", "design", "architecture", "strategy", "group_chat"],
```

- [ ] **Step 5: Add autogen config to `aura.config.json`**

Add after the `"github_connector"` section:

```json
"autogen": {
    "enabled": true,
    "agents": [
        {"name": "ideator", "system_message": "Generate creative solutions"},
        {"name": "critic", "system_message": "Find risks and flaws"},
        {"name": "synthesizer", "system_message": "Merge ideas into actionable plan"}
    ],
    "max_turns": 6,
    "llm_config": {}
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_autogen_registration.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 7: Commit**

```bash
git add agents/registry.py orchestrator_hub/router.py aura.config.json tests/test_autogen_registration.py
git commit -m "feat: register AutoGen agent in registry and router"
```

---

## Task 5: Optimized Multi-Stage Dockerfile

**Files:**
- Modify: `Dockerfile`

Note: This task is infrastructure — no TDD component. Verification is via `docker build`.

- [ ] **Step 1: Write the updated Dockerfile**

Replace the existing `Dockerfile` with the optimized multi-stage build:

```dockerfile
# Stage 1: Builder
FROM python:3.12-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Stage 2: Runtime
FROM python:3.12-slim AS runtime
WORKDIR /app
RUN groupadd -r aura && useradd -r -g aura -s /sbin/nologin aura
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/aura_cli /app/aura_cli
COPY --from=builder /app/core /app/core
COPY --from=builder /app/agents /app/agents
COPY --from=builder /app/memory /app/memory
COPY --from=builder /app/tools /app/tools
COPY --from=builder /app/orchestrator_hub /app/orchestrator_hub
COPY --from=builder /app/main.py /app/main.py
COPY --from=builder /app/run_aura.sh /app/run_aura.sh
COPY --from=builder /app/aura.config.json /app/aura.config.json
RUN chown -R aura:aura /app && chmod +x /app/run_aura.sh
USER aura
ENV PYTHONPATH=/app
ENV PATH=/home/aura/.local/bin:$PATH
EXPOSE 8000
ENTRYPOINT ["./run_aura.sh"]
CMD ["--help"]
```

- [ ] **Step 2: Verify Docker build succeeds**

Run: `docker build -t aura-cli:test . 2>&1 | tail -5`
Expected: `Successfully tagged aura-cli:test`

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "build: optimize Dockerfile with multi-stage build and non-root user"
```

---

## Task 6: CI/CD Workflow Update

**Files:**
- Modify: `.github/workflows/ci.yml`

Note: This task is infrastructure — no TDD component. Verification is via CI run after push.

- [ ] **Step 1: Add migration assistant test to CI**

In the `cli_docs_and_help_contracts` job, add a step after the existing test steps:

```yaml
    - name: Verify migration assistant tests
      run: |
        pytest -q tests/test_migration_assistant.py
```

- [ ] **Step 2: Add type check step to lint job (advisory)**

Add a new step to the `lint` job, after the ruff steps:

```yaml
    - name: Type check (advisory)
      continue-on-error: true
      run: mypy aura_cli/ core/ --ignore-missing-imports --no-error-summary
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add migration assistant tests and type checking to CI"
```

---

## Task 7: Integration Verification

**Files:**
- No new files — this is a verification task

- [ ] **Step 1: Run full test suite**

Run: `python3 -m pytest -q --ignore=tests/integration 2>&1 | tail -10`
Expected: All existing tests still pass, new tests pass

- [ ] **Step 2: Run lint**

Run: `ruff check . && ruff format --check .`
Expected: No errors

- [ ] **Step 3: Verify CLI still works**

Run: `python3 main.py --help && python3 main.py doctor`
Expected: Both succeed with expected output

- [ ] **Step 4: Verify AutoGen agent loads**

Run: `python3 -c "from agents.autogen_agent import AutoGenGroupChatAgent; a = AutoGenGroupChatAgent(); print(a.name, a._autogen_available)"`
Expected: `autogen_group_chat True`

- [ ] **Step 5: Update snapshots if needed**

Run: `python3 scripts/generate_cli_reference.py`
If any snapshots changed, commit them:

```bash
git add tests/snapshots/ docs/CLI_REFERENCE.md
git commit -m "chore: update snapshots after migration assistant integration"
```

---

## Verification Plan

After all tasks complete:

1. **New tests:** `python3 -m pytest tests/test_migration_assistant.py tests/test_autogen_agent.py tests/test_autogen_registration.py tests/test_deps_available.py -v`
2. **Full test suite:** `python3 -m pytest -q`
3. **Lint:** `ruff check .`
4. **CLI smoke test:** `python3 main.py doctor`
5. **Docker build:** `docker build -t aura-cli:test .`
6. **Import check:** `python3 -c "from aura_cli.migration_assistant import MigrationPrompts; print(len(MigrationPrompts.get_all_migration_prompts()))"`
