# Agent SDK Closed-Loop Meta-Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace AURA's rigid phase-sequenced orchestrator with a Claude Agent SDK-powered meta-controller that uses Claude as the decision-making brain — dynamically selecting workflows, tools, MCP servers, skills, and agents based on goal context and developmental task requirements.

**Architecture:** The meta-controller exposes AURA's entire infrastructure (30+ skills, 7 MCP servers, 10+ agents, workflow engine, memory system) as custom MCP tools to a Claude Agent SDK session. Claude decides what to invoke, when, and in what order — replacing hardcoded phase dispatch with context-aware reasoning. Subagents handle parallel workstreams; hooks enforce quality gates and capture feedback for the reflection loop.

**Tech Stack:** `claude-agent-sdk` (Python), `anthropic` SDK, existing AURA infrastructure (`core/`, `agents/`, `tools/`, `memory/`), `anyio` for async, `pydantic` for schemas.

---

## File Structure

```
core/agent_sdk/                    # New package — Agent SDK meta-controller
├── __init__.py                    # Package exports
├── config.py                      # Configuration: models, limits, tool selection
├── context_builder.py             # Build rich context from goal + memory + project state
├── tool_registry.py               # Custom MCP tools wrapping AURA infrastructure
├── subagent_definitions.py        # Subagent specs for parallel execution
├── hooks.py                       # Quality gate, logging, and feedback hooks
├── controller.py                  # Main meta-controller entry point
└── cli_integration.py             # Wire into aura_cli commands

tests/
├── test_agent_sdk_config.py
├── test_agent_sdk_context_builder.py
├── test_agent_sdk_tool_registry.py
├── test_agent_sdk_subagents.py
├── test_agent_sdk_hooks.py
├── test_agent_sdk_controller.py
└── test_agent_sdk_cli_integration.py
```

**Responsibilities:**

| File | Single Responsibility |
|------|----------------------|
| `config.py` | All configuration: model IDs, token limits, permission modes, tool allowlists |
| `context_builder.py` | Assemble goal context from memory, skills, project state — no LLM calls |
| `tool_registry.py` | Define and register all custom MCP tools that wrap AURA infrastructure |
| `subagent_definitions.py` | Define subagent types with their tool access and system prompts |
| `hooks.py` | Pre/post tool hooks for validation, logging, quality gates, metrics |
| `controller.py` | Orchestrate: build context → configure SDK → run query → handle result |
| `cli_integration.py` | Wire controller into `aura_cli` command dispatch |

---

## Task 1: Configuration Module

**Files:**
- Create: `core/agent_sdk/__init__.py`
- Create: `core/agent_sdk/config.py`
- Test: `tests/test_agent_sdk_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_sdk_config.py
"""Tests for Agent SDK configuration."""
import unittest
from unittest.mock import patch
import os


class TestAgentSDKConfig(unittest.TestCase):
    """Test Agent SDK configuration loading and defaults."""

    def test_default_config_has_required_fields(self):
        from core.agent_sdk.config import AgentSDKConfig

        config = AgentSDKConfig()
        self.assertEqual(config.model, "claude-sonnet-4-6")
        self.assertIsInstance(config.max_turns, int)
        self.assertGreater(config.max_turns, 0)
        self.assertIsInstance(config.max_budget_usd, float)
        self.assertIn(config.permission_mode, ("default", "acceptEdits", "bypassPermissions", "plan"))

    def test_config_from_aura_config(self):
        from core.agent_sdk.config import AgentSDKConfig

        aura_config = {
            "agent_sdk": {
                "model": "claude-opus-4-6",
                "max_turns": 50,
                "max_budget_usd": 5.0,
                "permission_mode": "acceptEdits",
            }
        }
        config = AgentSDKConfig.from_aura_config(aura_config)
        self.assertEqual(config.model, "claude-opus-4-6")
        self.assertEqual(config.max_turns, 50)
        self.assertEqual(config.max_budget_usd, 5.0)

    def test_config_tool_allowlist_defaults(self):
        from core.agent_sdk.config import AgentSDKConfig

        config = AgentSDKConfig()
        self.assertIn("Read", config.allowed_tools)
        self.assertIn("Edit", config.allowed_tools)
        self.assertIn("Bash", config.allowed_tools)
        self.assertIn("Glob", config.allowed_tools)
        self.assertIn("Grep", config.allowed_tools)
        self.assertIn("Agent", config.allowed_tools)

    def test_config_env_override(self):
        from core.agent_sdk.config import AgentSDKConfig

        with patch.dict(os.environ, {"AURA_AGENT_SDK_MODEL": "claude-opus-4-6"}):
            config = AgentSDKConfig()
            config.apply_env_overrides()
            self.assertEqual(config.model, "claude-opus-4-6")

    def test_config_mcp_server_endpoints(self):
        from core.agent_sdk.config import AgentSDKConfig

        config = AgentSDKConfig()
        endpoints = config.mcp_server_endpoints
        self.assertIsInstance(endpoints, dict)
        # Should include known AURA MCP servers
        self.assertIn("dev_tools", endpoints)
        self.assertIn("skills", endpoints)
        self.assertIn("control", endpoints)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agent_sdk_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.agent_sdk'`

- [ ] **Step 3: Create package init**

```python
# core/agent_sdk/__init__.py
"""Claude Agent SDK meta-controller for AURA CLI.

Replaces rigid phase-sequenced orchestration with Claude-as-brain,
dynamically selecting workflows, tools, MCP servers, and skills
based on goal context.
"""
from core.agent_sdk.config import AgentSDKConfig

__all__ = ["AgentSDKConfig"]
```

- [ ] **Step 4: Write minimal implementation**

```python
# core/agent_sdk/config.py
"""Configuration for the Agent SDK meta-controller."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Default MCP server ports matching aura.config.json
_DEFAULT_MCP_PORTS: Dict[str, int] = {
    "dev_tools": 8001,
    "skills": 8002,
    "control": 8003,
    "thinking": 8004,
    "agentic_loop": 8006,
    "copilot": 8007,
    "sadd": 8020,
    "discovery": 8025,
}

# Built-in tools from Agent SDK that we always allow
_DEFAULT_ALLOWED_TOOLS: List[str] = [
    "Read", "Write", "Edit", "Bash", "Glob", "Grep", "Agent",
]


@dataclass
class AgentSDKConfig:
    """Configuration for Agent SDK meta-controller sessions."""

    model: str = "claude-sonnet-4-6"
    max_turns: int = 30
    max_budget_usd: float = 2.0
    permission_mode: str = "acceptEdits"
    allowed_tools: List[str] = field(default_factory=lambda: list(_DEFAULT_ALLOWED_TOOLS))
    mcp_ports: Dict[str, int] = field(default_factory=lambda: dict(_DEFAULT_MCP_PORTS))
    project_root: Optional[str] = None
    enable_thinking: bool = True
    enable_subagents: bool = True
    enable_hooks: bool = True

    @classmethod
    def from_aura_config(cls, aura_config: Dict[str, Any]) -> "AgentSDKConfig":
        """Build config from aura.config.json dict."""
        sdk_section = aura_config.get("agent_sdk", {})
        mcp_ports = aura_config.get("mcp_servers", dict(_DEFAULT_MCP_PORTS))
        return cls(
            model=sdk_section.get("model", cls.model),
            max_turns=sdk_section.get("max_turns", cls.max_turns),
            max_budget_usd=sdk_section.get("max_budget_usd", cls.max_budget_usd),
            permission_mode=sdk_section.get("permission_mode", cls.permission_mode),
            allowed_tools=sdk_section.get("allowed_tools", list(_DEFAULT_ALLOWED_TOOLS)),
            mcp_ports=mcp_ports,
            project_root=sdk_section.get("project_root"),
            enable_thinking=sdk_section.get("enable_thinking", True),
            enable_subagents=sdk_section.get("enable_subagents", True),
            enable_hooks=sdk_section.get("enable_hooks", True),
        )

    def apply_env_overrides(self) -> None:
        """Override config values from environment variables."""
        if model := os.environ.get("AURA_AGENT_SDK_MODEL"):
            self.model = model
        if max_turns := os.environ.get("AURA_AGENT_SDK_MAX_TURNS"):
            self.max_turns = int(max_turns)
        if budget := os.environ.get("AURA_AGENT_SDK_MAX_BUDGET"):
            self.max_budget_usd = float(budget)
        if mode := os.environ.get("AURA_AGENT_SDK_PERMISSION_MODE"):
            self.permission_mode = mode

    @property
    def mcp_server_endpoints(self) -> Dict[str, str]:
        """Return {name: url} for all configured MCP servers."""
        return {
            name: f"http://localhost:{port}"
            for name, port in self.mcp_ports.items()
        }

    @property
    def thinking_config(self) -> Optional[Dict[str, str]]:
        """Return thinking config for Agent SDK options."""
        if self.enable_thinking:
            return {"type": "adaptive"}
        return None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_agent_sdk_config.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 6: Commit**

```bash
git add core/agent_sdk/__init__.py core/agent_sdk/config.py tests/test_agent_sdk_config.py
git commit -m "feat: add Agent SDK config module with defaults and env overrides"
```

---

## Task 2: Context Builder

**Files:**
- Create: `core/agent_sdk/context_builder.py`
- Test: `tests/test_agent_sdk_context_builder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_sdk_context_builder.py
"""Tests for Agent SDK context builder."""
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestContextBuilder(unittest.TestCase):
    """Test context assembly from AURA subsystems."""

    def test_build_basic_context(self):
        from core.agent_sdk.context_builder import ContextBuilder

        builder = ContextBuilder(project_root=Path("/tmp/test-project"))
        ctx = builder.build(goal="Add unit tests for auth module")
        self.assertIn("goal", ctx)
        self.assertEqual(ctx["goal"], "Add unit tests for auth module")
        self.assertIn("goal_type", ctx)
        self.assertIn("project_root", ctx)

    def test_goal_classification(self):
        from core.agent_sdk.context_builder import ContextBuilder

        builder = ContextBuilder(project_root=Path("/tmp/test-project"))
        self.assertEqual(builder.classify_goal("Fix null pointer in login"), "bug_fix")
        self.assertEqual(builder.classify_goal("Add OAuth2 support"), "feature")
        self.assertEqual(builder.classify_goal("Extract helper methods"), "refactor")
        self.assertEqual(builder.classify_goal("Scan for SQL injection"), "security")

    def test_skill_recommendations(self):
        from core.agent_sdk.context_builder import ContextBuilder

        builder = ContextBuilder(project_root=Path("/tmp/test-project"))
        ctx = builder.build(goal="Fix authentication bug")
        self.assertIn("recommended_skills", ctx)
        self.assertIsInstance(ctx["recommended_skills"], list)
        self.assertGreater(len(ctx["recommended_skills"]), 0)

    def test_memory_hints_with_brain(self):
        from core.agent_sdk.context_builder import ContextBuilder

        mock_brain = MagicMock()
        mock_brain.recall_with_budget.return_value = [
            {"content": "Auth uses JWT tokens", "score": 0.9}
        ]
        builder = ContextBuilder(
            project_root=Path("/tmp/test-project"),
            brain=mock_brain,
        )
        ctx = builder.build(goal="Fix auth token expiry")
        self.assertIn("memory_hints", ctx)
        self.assertGreater(len(ctx["memory_hints"]), 0)

    def test_memory_hints_without_brain(self):
        from core.agent_sdk.context_builder import ContextBuilder

        builder = ContextBuilder(project_root=Path("/tmp/test-project"))
        ctx = builder.build(goal="Fix auth bug")
        self.assertIn("memory_hints", ctx)
        self.assertEqual(ctx["memory_hints"], [])

    def test_mcp_tool_summary(self):
        from core.agent_sdk.context_builder import ContextBuilder

        builder = ContextBuilder(project_root=Path("/tmp/test-project"))
        ctx = builder.build(goal="Lint and format codebase")
        self.assertIn("available_mcp_categories", ctx)
        self.assertIsInstance(ctx["available_mcp_categories"], list)

    def test_build_system_prompt(self):
        from core.agent_sdk.context_builder import ContextBuilder

        builder = ContextBuilder(project_root=Path("/tmp/test-project"))
        prompt = builder.build_system_prompt(goal="Add tests", goal_type="feature")
        self.assertIn("AURA", prompt)
        self.assertIn("feature", prompt)
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 100)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agent_sdk_context_builder.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# core/agent_sdk/context_builder.py
"""Build rich context for Agent SDK sessions from AURA subsystems.

Assembles goal classification, memory hints, skill recommendations,
and MCP tool availability — without making any LLM calls.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

# Reuse existing AURA goal classification
from core.skill_dispatcher import SKILL_MAP, _GOAL_TYPE_HINTS


# MCP server categories for tool routing hints
_MCP_CATEGORIES: Dict[str, List[str]] = {
    "code_analysis": ["dev_tools", "skills"],
    "project_management": ["control", "sadd"],
    "reasoning": ["thinking"],
    "orchestration": ["agentic_loop"],
    "github": ["copilot"],
}

_SYSTEM_PROMPT_TEMPLATE = """\
You are the AURA Meta-Controller — an autonomous software development agent.

## Your Mission
Execute the following goal by intelligently selecting and composing tools, \
skills, agents, workflows, and MCP servers.

## Goal
{goal}

## Goal Type
{goal_type}

## Available Infrastructure

### Custom Tools (via MCP)
You have access to AURA's full infrastructure as custom tools:
- **analyze_goal**: Classify goals and gather project context
- **dispatch_skills**: Run static analysis skills (30+ available) matched to goal type
- **create_plan**: Generate step-by-step implementation plans
- **critique_plan**: Adversarial review of plans for gaps and risks
- **synthesize_task**: Merge plan + critique into executable task bundle
- **generate_code**: Produce code changes with CoT reasoning
- **run_sandbox**: Execute code in isolated subprocess
- **apply_changes**: Write file changes atomically with overwrite safety
- **verify_changes**: Run tests and linters against changes
- **reflect_on_outcome**: Analyze cycle outcomes and update skill weights
- **search_memory**: Query semantic memory for relevant past context
- **store_memory**: Persist learnings for future cycles
- **manage_goals**: Add/list/complete/archive goals
- **discover_mcp_tools**: Find tools across all MCP servers
- **invoke_mcp_tool**: Call any tool on any MCP server
- **run_workflow**: Execute named workflow definitions

### Subagents
You can dispatch subagents for parallel work:
- **planning-agent**: Deep planning with full codebase access
- **implementation-agent**: Code generation with sandbox verification
- **verification-agent**: Comprehensive test/lint/security checks
- **research-agent**: Codebase exploration and context gathering

## Decision Framework
1. **Analyze first**: Always start by understanding the goal context
2. **Plan before acting**: Create and critique plans for non-trivial work
3. **Verify everything**: Run tests after every code change
4. **Reflect on outcomes**: Learn from successes and failures
5. **Use the right tool**: Match tools to the task — don't over-engineer

{context_section}

## Constraints
- Apply file changes atomically — never leave partial writes
- Run verification after every apply
- Store learnings in memory after each significant outcome
- Respect the project's overwrite safety policy for stale snippets
"""


class ContextBuilder:
    """Assemble goal context from AURA subsystems without LLM calls."""

    def __init__(
        self,
        project_root: Path,
        brain: Any = None,
        vector_store: Any = None,
    ) -> None:
        self._project_root = project_root
        self._brain = brain
        self._vector_store = vector_store

    def classify_goal(self, goal: str) -> str:
        """Classify goal using keyword-overlap scoring (no LLM call).

        Reuses the same scoring logic as ``core.skill_dispatcher.classify_goal``
        — counts keyword hits per goal type and picks the best match.
        """
        goal_lower = goal.lower()
        scores = {
            gt: sum(1 for kw in kws if kw in goal_lower)
            for gt, kws in _GOAL_TYPE_HINTS.items()
        }
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "default"

    def _get_recommended_skills(self, goal_type: str) -> List[str]:
        """Get skill names recommended for this goal type."""
        return list(SKILL_MAP.get(goal_type, SKILL_MAP.get("default", [])))

    def _get_memory_hints(self, goal: str) -> List[Dict[str, Any]]:
        """Retrieve relevant memory hints from brain."""
        if self._brain is None:
            return []
        try:
            return self._brain.recall_with_budget(goal, budget=5)
        except Exception:
            return []

    def _get_available_mcp_categories(self) -> List[str]:
        """Return available MCP tool categories."""
        return list(_MCP_CATEGORIES.keys())

    def build(self, goal: str) -> Dict[str, Any]:
        """Build complete context dict for a goal."""
        goal_type = self.classify_goal(goal)
        return {
            "goal": goal,
            "goal_type": goal_type,
            "project_root": str(self._project_root),
            "recommended_skills": self._get_recommended_skills(goal_type),
            "memory_hints": self._get_memory_hints(goal),
            "available_mcp_categories": self._get_available_mcp_categories(),
        }

    def build_system_prompt(self, goal: str, goal_type: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build the system prompt for the Agent SDK session."""
        context_section = ""
        if context:
            parts = []
            if context.get("memory_hints"):
                hints = "\n".join(
                    f"- {h.get('content', h) if isinstance(h, dict) else h}"
                    for h in context["memory_hints"]
                )
                parts.append(f"### Memory Hints\n{hints}")
            if context.get("recommended_skills"):
                skills = ", ".join(context["recommended_skills"])
                parts.append(f"### Recommended Skills\n{skills}")
            context_section = "\n\n".join(parts)

        return _SYSTEM_PROMPT_TEMPLATE.format(
            goal=goal,
            goal_type=goal_type,
            context_section=context_section,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_agent_sdk_context_builder.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 5: Commit**

```bash
git add core/agent_sdk/context_builder.py tests/test_agent_sdk_context_builder.py
git commit -m "feat: add context builder for Agent SDK goal analysis"
```

---

## Task 3: Custom MCP Tool Registry

**Files:**
- Create: `core/agent_sdk/tool_registry.py`
- Test: `tests/test_agent_sdk_tool_registry.py`

This is the largest and most critical module — it wraps AURA's entire infrastructure as MCP tools that Claude can invoke.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_sdk_tool_registry.py
"""Tests for Agent SDK custom MCP tool registry."""
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path


class TestToolRegistry(unittest.TestCase):
    """Test custom MCP tool creation wrapping AURA infrastructure."""

    def test_create_tools_returns_list(self):
        from core.agent_sdk.tool_registry import create_aura_tools

        tools = create_aura_tools(project_root=Path("/tmp/test"))
        self.assertIsInstance(tools, list)
        self.assertGreater(len(tools), 0)

    def test_all_required_tools_present(self):
        from core.agent_sdk.tool_registry import create_aura_tools

        tools = create_aura_tools(project_root=Path("/tmp/test"))
        tool_names = [t.name for t in tools]
        required = [
            "analyze_goal",
            "dispatch_skills",
            "create_plan",
            "critique_plan",
            "generate_code",
            "run_sandbox",
            "apply_changes",
            "verify_changes",
            "reflect_on_outcome",
            "search_memory",
            "store_memory",
            "manage_goals",
            "discover_mcp_tools",
            "invoke_mcp_tool",
            "run_workflow",
        ]
        for name in required:
            self.assertIn(name, tool_names, f"Missing required tool: {name}")

    def test_tools_have_descriptions(self):
        from core.agent_sdk.tool_registry import create_aura_tools

        tools = create_aura_tools(project_root=Path("/tmp/test"))
        for t in tools:
            self.assertTrue(
                len(t.description) > 10,
                f"Tool {t.name} has insufficient description",
            )

    def test_tools_have_input_schemas(self):
        from core.agent_sdk.tool_registry import create_aura_tools

        tools = create_aura_tools(project_root=Path("/tmp/test"))
        for t in tools:
            self.assertIsInstance(t.input_schema, dict, f"Tool {t.name} missing schema")
            self.assertIn("type", t.input_schema)


class TestAnalyzeGoalTool(unittest.TestCase):
    """Test the analyze_goal tool handler."""

    def test_analyze_goal_returns_context(self):
        from core.agent_sdk.tool_registry import _handle_analyze_goal

        result = _handle_analyze_goal(
            {"goal": "Fix login bug"},
            project_root=Path("/tmp/test"),
        )
        self.assertIn("goal_type", result)
        self.assertIn("recommended_skills", result)

    def test_analyze_goal_classifies_correctly(self):
        from core.agent_sdk.tool_registry import _handle_analyze_goal

        result = _handle_analyze_goal(
            {"goal": "Refactor the database layer"},
            project_root=Path("/tmp/test"),
        )
        self.assertEqual(result["goal_type"], "refactor")


class TestDispatchSkillsTool(unittest.TestCase):
    """Test the dispatch_skills tool handler."""

    def test_dispatch_with_no_skills_loaded(self):
        from core.agent_sdk.tool_registry import _handle_dispatch_skills

        result = _handle_dispatch_skills(
            {"goal_type": "bug_fix", "project_root": "/tmp/test"},
            project_root=Path("/tmp/test"),
        )
        self.assertIn("skills_run", result)
        self.assertIsInstance(result["skills_run"], list)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agent_sdk_tool_registry.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# core/agent_sdk/tool_registry.py
"""Custom MCP tools wrapping AURA infrastructure for Agent SDK.

Each tool bridges a Claude Agent SDK tool call to an existing AURA
subsystem (agents, skills, MCP servers, memory, workflows).
"""
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class AuraTool:
    """A custom tool definition for the Agent SDK MCP server."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[..., Any] = field(repr=False)


# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------

def _handle_analyze_goal(
    args: Dict[str, Any],
    *,
    project_root: Path,
    brain: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Classify goal and gather context."""
    from core.agent_sdk.context_builder import ContextBuilder

    builder = ContextBuilder(project_root=project_root, brain=brain)
    return builder.build(goal=args["goal"])


def _handle_dispatch_skills(
    args: Dict[str, Any],
    *,
    project_root: Path,
    brain: Any = None,
    model_adapter: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Run adaptive skills matched to goal type.

    dispatch_skills(goal_type, skills, project_root) expects skills
    as a Dict[str, SkillBase] (name → instance), not a list.
    """
    goal_type = args.get("goal_type", "default")
    root = Path(args.get("project_root", str(project_root)))

    try:
        from core.skill_dispatcher import dispatch_skills, SKILL_MAP
        from agents.skills.registry import all_skills

        skill_names = SKILL_MAP.get(goal_type, SKILL_MAP.get("default", []))
        available = all_skills(brain=brain, model=model_adapter)
        # Filter to a dict of only the relevant skills
        skills = {n: available[n] for n in skill_names if n in available}

        if not skills:
            return {"skills_run": [], "results": {}, "note": "No matching skills loaded"}

        results = dispatch_skills(goal_type, skills, str(root))
        return {
            "skills_run": list(skills.keys()),
            "results": results,
        }
    except Exception as exc:
        return {
            "skills_run": [],
            "results": {},
            "error": str(exc),
        }


def _handle_create_plan(
    args: Dict[str, Any],
    *,
    brain: Any = None,
    model_adapter: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Generate implementation plan via PlannerAgent."""
    try:
        from agents.planner import PlannerAgent

        agent = PlannerAgent(brain=brain, model=model_adapter)
        result = agent.plan(
            goal=args["goal"],
            memory_snapshot=args.get("memory_snapshot", ""),
            similar_past_problems=args.get("similar_past_problems", ""),
            known_weaknesses=args.get("known_weaknesses", ""),
        )
        if isinstance(result, list):
            return {"steps": result, "confidence": 0.7}
        return result
    except Exception as exc:
        return {"steps": [], "error": str(exc)}


def _handle_critique_plan(
    args: Dict[str, Any],
    *,
    brain: Any = None,
    model_adapter: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Adversarial plan review via CriticAgent.

    CriticAgent.critique_plan() calls self.brain.recall_with_budget()
    internally, so brain must not be None.
    """
    if brain is None:
        return {"issues": [], "error": "CriticAgent requires a brain instance"}
    try:
        from agents.critic import CriticAgent

        agent = CriticAgent(brain=brain, model=model_adapter)
        result = agent.critique_plan(
            task=args["goal"],
            plan=args["plan"],
        )
        return result if isinstance(result, dict) else {"issues": [], "raw": str(result)}
    except Exception as exc:
        return {"issues": [], "error": str(exc)}


def _handle_generate_code(
    args: Dict[str, Any],
    *,
    brain: Any = None,
    model_adapter: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Generate code changes via CoderAgent.

    CoderAgent.implement(task) takes a single positional string.
    Combine task + plan + context into one prompt string.
    """
    try:
        from agents.coder import CoderAgent

        agent = CoderAgent(brain=brain, model=model_adapter)
        # Build a single task string — implement() takes one positional arg
        task_parts = [args["task"]]
        if args.get("plan"):
            task_parts.append(f"\n\nPlan:\n{args['plan']}")
        if args.get("context"):
            task_parts.append(f"\n\nContext:\n{args['context']}")
        result = agent.implement("\n".join(task_parts))
        if isinstance(result, str):
            return {"code": result, "confidence": 0.7}
        return result
    except Exception as exc:
        return {"code": "", "error": str(exc)}


def _handle_run_sandbox(
    args: Dict[str, Any],
    *,
    project_root: Path,
    brain: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Execute code in sandboxed subprocess.

    SandboxAgent.__init__(brain, timeout, python_exec) requires brain
    as a positional argument.
    """
    try:
        from agents.sandbox import SandboxAgent

        agent = SandboxAgent(brain=brain, timeout=args.get("timeout", 30))
        result = agent.run({
            "code": args["code"],
            "language": args.get("language", "python"),
            "project_root": str(project_root),
        })
        return result if isinstance(result, dict) else {"output": str(result)}
    except Exception as exc:
        return {"output": "", "error": str(exc), "exit_code": 1}


def _handle_apply_changes(
    args: Dict[str, Any],
    *,
    project_root: Path,
    **_: Any,
) -> Dict[str, Any]:
    """Apply file changes atomically."""
    try:
        from core.file_tools import apply_change_with_explicit_overwrite_policy

        result = apply_change_with_explicit_overwrite_policy(
            project_root=project_root,
            file_path=args["file_path"],
            new_code=args["new_code"],
            old_code=args.get("old_code", ""),
            overwrite_file=args.get("overwrite_file", False),
        )
        return {"applied": True, "result": str(result)}
    except Exception as exc:
        return {"applied": False, "error": str(exc)}


def _handle_verify_changes(
    args: Dict[str, Any],
    *,
    project_root: Path = None,
    **_: Any,
) -> Dict[str, Any]:
    """Run tests and linters via VerifierAgent.

    VerifierAgent.__init__(timeout) takes only an optional timeout.
    No brain or model_adapter params.
    """
    try:
        from agents.verifier import VerifierAgent

        agent = VerifierAgent(timeout=120)
        result = agent.run({
            "goal": args.get("goal", "verify changes"),
            "project_root": str(project_root or "."),
            "changed_files": args.get("changed_files", []),
        })
        return result if isinstance(result, dict) else {"passed": False, "raw": str(result)}
    except Exception as exc:
        return {"passed": False, "error": str(exc)}


def _handle_reflect(
    args: Dict[str, Any],
    **_: Any,
) -> Dict[str, Any]:
    """Reflect on outcome via ReflectorAgent.

    ReflectorAgent extends Agent (ABC) with no __init__ params.
    Its run(input_data) accepts a dict with verification + skill_context.
    """
    try:
        from agents.reflector import ReflectorAgent

        agent = ReflectorAgent()
        result = agent.run({
            "goal": args["goal"],
            "verification": args.get("verification", {}),
            "skill_context": args.get("skill_context", {}),
        })
        return result if isinstance(result, dict) else {"summary": str(result)}
    except Exception as exc:
        return {"summary": "", "error": str(exc)}


def _handle_search_memory(
    args: Dict[str, Any],
    *,
    brain: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Query semantic memory.

    Brain.recall_with_budget(max_tokens, tier) returns List[str] of
    memories truncated to fit within max_tokens. It does NOT accept
    a query string — it returns the most recent memories by budget.
    """
    if brain is None:
        return {"results": [], "note": "No brain configured"}
    try:
        max_tokens = args.get("max_tokens", 4000)
        results = brain.recall_with_budget(max_tokens=max_tokens)
        # Filter results by query keyword if provided
        query = args.get("query", "").lower()
        if query:
            results = [r for r in results if query in r.lower()] or results
        return {"results": results}
    except Exception as exc:
        return {"results": [], "error": str(exc)}


def _handle_store_memory(
    args: Dict[str, Any],
    *,
    brain: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Store a memory in brain.

    Brain.remember(data) takes a single argument — either a string
    or a dict (dict is JSON-serialized internally).
    """
    if brain is None:
        return {"stored": False, "note": "No brain configured"}
    try:
        content = args["content"]
        tags = args.get("tags", [])
        # Pack tags into a dict if provided, else store raw string
        if tags:
            brain.remember({"content": content, "tags": tags})
        else:
            brain.remember(content)
        return {"stored": True}
    except Exception as exc:
        return {"stored": False, "error": str(exc)}


def _handle_manage_goals(
    args: Dict[str, Any],
    *,
    goal_queue: Any = None,
    goal_archive: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Manage the goal queue (add/list/complete/archive)."""
    action = args.get("action", "list")
    try:
        if goal_queue is None:
            from core.goal_queue import GoalQueue
            goal_queue = GoalQueue()

        if action == "add":
            goal_queue.add(args["goal"])
            return {"action": "add", "success": True}
        elif action == "list":
            return {"action": "list", "goals": list(goal_queue._queue)}
        elif action == "complete":
            goal_queue.complete(args["goal"])
            return {"action": "complete", "success": True}
        elif action == "next":
            g = goal_queue.next()
            return {"action": "next", "goal": g}
        else:
            return {"error": f"Unknown action: {action}"}
    except Exception as exc:
        return {"error": str(exc)}


def _handle_discover_mcp_tools(
    args: Dict[str, Any],
    *,
    config: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Discover tools across MCP servers."""
    try:
        import requests

        discovery_url = "http://localhost:8025/call"
        resp = requests.post(
            discovery_url,
            json={"tool_name": "search_tools_semantically", "args": {"query": args.get("query", "")}},
            timeout=3,
        )
        if resp.ok:
            return resp.json()
        return {"tools": [], "error": f"Discovery server returned {resp.status_code}"}
    except Exception as exc:
        return {"tools": [], "error": str(exc)}


def _handle_invoke_mcp_tool(
    args: Dict[str, Any],
    *,
    config: Any = None,
    **_: Any,
) -> Dict[str, Any]:
    """Invoke a tool on any MCP server."""
    try:
        import requests

        server = args["server"]
        from core.agent_sdk.config import AgentSDKConfig

        cfg = config or AgentSDKConfig()
        port = cfg.mcp_ports.get(server)
        if not port:
            return {"error": f"Unknown MCP server: {server}"}

        resp = requests.post(
            f"http://localhost:{port}/call",
            json={"tool_name": args["tool_name"], "args": args.get("tool_args", {})},
            timeout=args.get("timeout", 30),
        )
        return resp.json() if resp.ok else {"error": f"Server returned {resp.status_code}"}
    except Exception as exc:
        return {"error": str(exc)}


def _handle_run_workflow(
    args: Dict[str, Any],
    **_: Any,
) -> Dict[str, Any]:
    """Execute a named workflow definition."""
    try:
        from core.workflow_engine import WorkflowEngine

        engine = WorkflowEngine()
        result = engine.execute(
            workflow_name=args["workflow_name"],
            inputs=args.get("inputs", {}),
        )
        return result if isinstance(result, dict) else {"result": str(result)}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------

_TOOL_DEFS: List[Dict[str, Any]] = [
    {
        "name": "analyze_goal",
        "description": "Classify a development goal and gather project context including goal type, recommended skills, memory hints, and available MCP tool categories. Always call this first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "The development goal to analyze"},
            },
            "required": ["goal"],
        },
        "handler": _handle_analyze_goal,
    },
    {
        "name": "dispatch_skills",
        "description": "Run static analysis skills matched to a goal type. Returns findings from skills like linter, type checker, complexity scorer, security scanner, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "goal_type": {"type": "string", "enum": ["bug_fix", "feature", "refactor", "security", "docs", "default"]},
                "project_root": {"type": "string", "description": "Project root path (optional, uses config default)"},
            },
            "required": ["goal_type"],
        },
        "handler": _handle_dispatch_skills,
    },
    {
        "name": "create_plan",
        "description": "Generate a step-by-step implementation plan using the PlannerAgent (Senior Software Architect). Returns structured plan with steps, confidence, and complexity assessment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "What to accomplish"},
                "memory_snapshot": {"type": "string", "description": "Current memory context snapshot"},
                "similar_past_problems": {"type": "string", "description": "Relevant past problems from memory"},
                "known_weaknesses": {"type": "string", "description": "Known system weaknesses to address"},
            },
            "required": ["goal"],
        },
        "handler": _handle_create_plan,
    },
    {
        "name": "critique_plan",
        "description": "Adversarial review of a plan by the CriticAgent (Principal Engineer). Checks correctness, security, completeness, feasibility. Returns issues with severity levels.",
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "The original goal"},
                "plan": {"type": "array", "items": {"type": "string"}, "description": "Plan steps to critique"},
            },
            "required": ["goal", "plan"],
        },
        "handler": _handle_critique_plan,
    },
    {
        "name": "generate_code",
        "description": "Generate code changes via the CoderAgent (Expert Python Developer). Uses CoT reasoning for problem analysis, approach selection, and testing strategy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "What code to write or change"},
                "plan": {"type": "string", "description": "Implementation plan to follow"},
                "context": {"type": "string", "description": "Relevant code context"},
            },
            "required": ["task"],
        },
        "handler": _handle_generate_code,
    },
    {
        "name": "run_sandbox",
        "description": "Execute code in an isolated subprocess sandbox. Returns stdout, stderr, and exit code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Code to execute"},
                "language": {"type": "string", "default": "python"},
            },
            "required": ["code"],
        },
        "handler": _handle_run_sandbox,
    },
    {
        "name": "apply_changes",
        "description": "Apply file changes atomically with overwrite safety. Respects AURA's stale-snippet mismatch policy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Relative path within project"},
                "new_code": {"type": "string", "description": "New file content or code block"},
                "old_code": {"type": "string", "description": "Expected existing content (empty for new file)"},
                "overwrite_file": {"type": "boolean", "default": False},
            },
            "required": ["file_path", "new_code"],
        },
        "handler": _handle_apply_changes,
    },
    {
        "name": "verify_changes",
        "description": "Run tests and linters against recent changes via VerifierAgent. Returns pass/fail status with details.",
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "What was being worked on"},
                "changed_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of changed file paths",
                },
            },
            "required": ["goal"],
        },
        "handler": _handle_verify_changes,
    },
    {
        "name": "reflect_on_outcome",
        "description": "Analyze the outcome of a development cycle. Updates skill weights, records learnings, detects patterns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "verification": {"type": "object", "description": "Verification results from verify_changes"},
                "skill_context": {"type": "object", "description": "Skill execution results from dispatch_skills"},
            },
            "required": ["goal"],
        },
        "handler": _handle_reflect,
    },
    {
        "name": "search_memory",
        "description": "Retrieve recent memories from AURA's brain, optionally filtered by a keyword query. Returns memories truncated to fit within max_tokens.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Optional keyword to filter results by"},
                "max_tokens": {"type": "integer", "default": 4000, "description": "Token budget for returned memories"},
            },
        },
        "handler": _handle_search_memory,
    },
    {
        "name": "store_memory",
        "description": "Persist a learning, decision, or insight to AURA's semantic memory for future recall.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "What to remember"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for retrieval"},
            },
            "required": ["content"],
        },
        "handler": _handle_store_memory,
    },
    {
        "name": "manage_goals",
        "description": "Manage the persistent goal queue: add new goals, list pending, complete, or get next.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "list", "complete", "next"]},
                "goal": {"type": "string", "description": "Goal text (for add/complete actions)"},
            },
            "required": ["action"],
        },
        "handler": _handle_manage_goals,
    },
    {
        "name": "discover_mcp_tools",
        "description": "Search for available tools across all AURA MCP servers (dev_tools, skills, control, thinking, agentic_loop, copilot, sadd).",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What kind of tool you need"},
            },
            "required": ["query"],
        },
        "handler": _handle_discover_mcp_tools,
    },
    {
        "name": "invoke_mcp_tool",
        "description": "Call any tool on any AURA MCP server by name. Use discover_mcp_tools first to find available tools.",
        "input_schema": {
            "type": "object",
            "properties": {
                "server": {"type": "string", "enum": ["dev_tools", "skills", "control", "thinking", "agentic_loop", "copilot", "sadd"]},
                "tool_name": {"type": "string", "description": "Tool name on the server"},
                "tool_args": {"type": "object", "description": "Arguments for the tool"},
                "timeout": {"type": "integer", "default": 30},
            },
            "required": ["server", "tool_name"],
        },
        "handler": _handle_invoke_mcp_tool,
    },
    {
        "name": "run_workflow",
        "description": "Execute a named workflow definition from the workflow engine. Workflows are DAG-based multi-step pipelines.",
        "input_schema": {
            "type": "object",
            "properties": {
                "workflow_name": {"type": "string", "description": "Name of the workflow to run"},
                "inputs": {"type": "object", "description": "Input parameters for the workflow"},
            },
            "required": ["workflow_name"],
        },
        "handler": _handle_run_workflow,
    },
]


def create_aura_tools(
    project_root: Path,
    brain: Any = None,
    model_adapter: Any = None,
    goal_queue: Any = None,
    goal_archive: Any = None,
    config: Any = None,
) -> List[AuraTool]:
    """Create all custom AURA tools for Agent SDK registration."""
    deps = dict(
        project_root=project_root,
        brain=brain,
        model_adapter=model_adapter,
        goal_queue=goal_queue,
        goal_archive=goal_archive,
        config=config,
    )

    tools = []
    for defn in _TOOL_DEFS:
        raw_handler = defn["handler"]

        def make_bound(h: Callable, d: Dict) -> Callable:
            def bound(args: Dict[str, Any]) -> Any:
                return h(args, **d)
            return bound

        tools.append(AuraTool(
            name=defn["name"],
            description=defn["description"],
            input_schema=defn["input_schema"],
            handler=make_bound(raw_handler, deps),
        ))
    return tools
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_agent_sdk_tool_registry.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add core/agent_sdk/tool_registry.py tests/test_agent_sdk_tool_registry.py
git commit -m "feat: add custom MCP tool registry wrapping AURA infrastructure"
```

---

## Task 4: Subagent Definitions

**Files:**
- Create: `core/agent_sdk/subagent_definitions.py`
- Test: `tests/test_agent_sdk_subagents.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_sdk_subagents.py
"""Tests for Agent SDK subagent definitions."""
import unittest


class TestSubagentDefinitions(unittest.TestCase):
    """Test subagent creation for parallel work dispatch."""

    def test_get_all_subagents(self):
        from core.agent_sdk.subagent_definitions import get_subagent_definitions

        agents = get_subagent_definitions()
        self.assertIsInstance(agents, dict)
        self.assertGreater(len(agents), 0)

    def test_required_subagents_present(self):
        from core.agent_sdk.subagent_definitions import get_subagent_definitions

        agents = get_subagent_definitions()
        required = ["planning-agent", "implementation-agent", "verification-agent", "research-agent"]
        for name in required:
            self.assertIn(name, agents, f"Missing subagent: {name}")

    def test_subagent_has_description(self):
        from core.agent_sdk.subagent_definitions import get_subagent_definitions

        for name, defn in get_subagent_definitions().items():
            self.assertTrue(
                len(defn.description) > 20,
                f"Subagent {name} needs a better description",
            )

    def test_subagent_has_tools(self):
        from core.agent_sdk.subagent_definitions import get_subagent_definitions

        for name, defn in get_subagent_definitions().items():
            self.assertIsInstance(defn.tools, list)
            self.assertGreater(len(defn.tools), 0, f"Subagent {name} has no tools")

    def test_subagent_has_prompt(self):
        from core.agent_sdk.subagent_definitions import get_subagent_definitions

        for name, defn in get_subagent_definitions().items():
            self.assertTrue(
                len(defn.prompt) > 20,
                f"Subagent {name} needs a better prompt",
            )

    def test_get_subagent_for_task_type(self):
        from core.agent_sdk.subagent_definitions import get_subagent_for_task

        agent = get_subagent_for_task("plan")
        self.assertEqual(agent, "planning-agent")

        agent = get_subagent_for_task("implement")
        self.assertEqual(agent, "implementation-agent")

        agent = get_subagent_for_task("verify")
        self.assertEqual(agent, "verification-agent")

        agent = get_subagent_for_task("research")
        self.assertEqual(agent, "research-agent")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agent_sdk_subagents.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# core/agent_sdk/subagent_definitions.py
"""Subagent definitions for parallel task dispatch via Agent SDK.

Each subagent is specialized for a class of work and gets a focused
tool set and system prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SubagentDef:
    """Definition for a subagent dispatched by the meta-controller."""

    description: str
    prompt: str
    tools: List[str]
    model: Optional[str] = None


# Task type → subagent name mapping
_TASK_TYPE_MAP: Dict[str, str] = {
    "plan": "planning-agent",
    "planning": "planning-agent",
    "design": "planning-agent",
    "implement": "implementation-agent",
    "code": "implementation-agent",
    "coding": "implementation-agent",
    "build": "implementation-agent",
    "verify": "verification-agent",
    "test": "verification-agent",
    "lint": "verification-agent",
    "security": "verification-agent",
    "research": "research-agent",
    "explore": "research-agent",
    "investigate": "research-agent",
    "analyze": "research-agent",
}


def get_subagent_definitions() -> Dict[str, SubagentDef]:
    """Return all available subagent definitions."""
    return {
        "planning-agent": SubagentDef(
            description=(
                "Senior Software Architect agent for deep planning. "
                "Analyzes codebases, identifies risks, decomposes complex goals "
                "into ordered implementation steps with dependency tracking."
            ),
            prompt=(
                "You are a Senior Software Architect planning an implementation. "
                "Read the codebase to understand the current architecture. "
                "Produce a detailed, ordered plan with clear steps, file targets, "
                "risk assessments, and verification criteria. "
                "Consider edge cases, backward compatibility, and test strategy."
            ),
            tools=["Read", "Glob", "Grep", "Bash"],
        ),
        "implementation-agent": SubagentDef(
            description=(
                "Expert developer agent for code generation. "
                "Writes clean, tested code following project conventions. "
                "Handles file creation, editing, and sandbox verification."
            ),
            prompt=(
                "You are an Expert Python Developer implementing a specific task. "
                "Read existing code to match conventions. Write clean code with "
                "type hints. Create or update tests alongside implementation. "
                "Run tests after every change to verify correctness. "
                "Use atomic file operations — never leave partial writes."
            ),
            tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        ),
        "verification-agent": SubagentDef(
            description=(
                "Quality assurance agent for comprehensive verification. "
                "Runs tests, linters, type checks, and security scans. "
                "Reports findings with severity and fix suggestions."
            ),
            prompt=(
                "You are a Principal QA Engineer verifying code changes. "
                "Run the full test suite. Check for lint violations, type errors, "
                "and security issues. Report every finding with severity level "
                "and a specific fix suggestion. Be thorough — miss nothing."
            ),
            tools=["Read", "Bash", "Glob", "Grep"],
        ),
        "research-agent": SubagentDef(
            description=(
                "Codebase research agent for exploration and context gathering. "
                "Explores architecture, traces call chains, identifies patterns, "
                "and summarizes findings for other agents."
            ),
            prompt=(
                "You are a codebase researcher gathering context for a development task. "
                "Explore the relevant parts of the codebase. Trace call chains, "
                "identify patterns, find related code, and summarize your findings "
                "concisely. Focus on what's relevant to the task at hand."
            ),
            tools=["Read", "Glob", "Grep", "Bash"],
        ),
    }


def get_subagent_for_task(task_type: str) -> Optional[str]:
    """Return the best subagent name for a task type, or None."""
    return _TASK_TYPE_MAP.get(task_type.lower())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_agent_sdk_subagents.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add core/agent_sdk/subagent_definitions.py tests/test_agent_sdk_subagents.py
git commit -m "feat: add subagent definitions for parallel task dispatch"
```

---

## Task 5: Hooks for Quality Gates and Feedback

**Files:**
- Create: `core/agent_sdk/hooks.py`
- Test: `tests/test_agent_sdk_hooks.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_sdk_hooks.py
"""Tests for Agent SDK hooks."""
import unittest
from unittest.mock import MagicMock
import time


class TestHookCreation(unittest.TestCase):
    """Test hook factory functions."""

    def test_create_hooks_returns_dict(self):
        from core.agent_sdk.hooks import create_hooks

        hooks = create_hooks()
        self.assertIsInstance(hooks, dict)

    def test_hooks_has_pre_tool_use(self):
        from core.agent_sdk.hooks import create_hooks

        hooks = create_hooks()
        self.assertIn("PreToolUse", hooks)

    def test_hooks_has_post_tool_use(self):
        from core.agent_sdk.hooks import create_hooks

        hooks = create_hooks()
        self.assertIn("PostToolUse", hooks)

    def test_hooks_has_stop(self):
        from core.agent_sdk.hooks import create_hooks

        hooks = create_hooks()
        self.assertIn("Stop", hooks)


class TestMetricsCollector(unittest.TestCase):
    """Test the metrics collector used by hooks."""

    def test_record_tool_call(self):
        from core.agent_sdk.hooks import MetricsCollector

        mc = MetricsCollector()
        mc.record_tool_call("create_plan", 1.5, success=True)
        stats = mc.get_stats()
        self.assertEqual(stats["tool_calls"]["create_plan"]["count"], 1)
        self.assertTrue(stats["tool_calls"]["create_plan"]["success_rate"] > 0)

    def test_record_multiple_calls(self):
        from core.agent_sdk.hooks import MetricsCollector

        mc = MetricsCollector()
        mc.record_tool_call("verify_changes", 2.0, success=True)
        mc.record_tool_call("verify_changes", 1.0, success=False)
        stats = mc.get_stats()
        self.assertEqual(stats["tool_calls"]["verify_changes"]["count"], 2)
        self.assertAlmostEqual(stats["tool_calls"]["verify_changes"]["success_rate"], 0.5)

    def test_get_summary(self):
        from core.agent_sdk.hooks import MetricsCollector

        mc = MetricsCollector()
        mc.record_tool_call("analyze_goal", 0.5, success=True)
        summary = mc.get_summary()
        self.assertIn("total_calls", summary)
        self.assertEqual(summary["total_calls"], 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agent_sdk_hooks.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# core/agent_sdk/hooks.py
"""Hooks for Agent SDK sessions: quality gates, logging, metrics.

Hooks intercept tool calls to enforce safety policies, collect metrics,
and feed the reflection loop.
"""
from __future__ import annotations

import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation."""

    tool_name: str
    elapsed_s: float
    success: bool
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Collect metrics from tool calls during a session."""

    def __init__(self) -> None:
        self._records: List[ToolCallRecord] = []

    def record_tool_call(self, tool_name: str, elapsed_s: float, success: bool) -> None:
        self._records.append(ToolCallRecord(
            tool_name=tool_name,
            elapsed_s=elapsed_s,
            success=success,
        ))

    def get_stats(self) -> Dict[str, Any]:
        """Per-tool aggregated statistics."""
        by_tool: Dict[str, List[ToolCallRecord]] = defaultdict(list)
        for r in self._records:
            by_tool[r.tool_name].append(r)

        stats: Dict[str, Any] = {"tool_calls": {}}
        for name, records in by_tool.items():
            successes = sum(1 for r in records if r.success)
            stats["tool_calls"][name] = {
                "count": len(records),
                "success_rate": successes / len(records) if records else 0,
                "avg_elapsed_s": sum(r.elapsed_s for r in records) / len(records),
            }
        return stats

    def get_summary(self) -> Dict[str, Any]:
        """High-level session summary."""
        successes = sum(1 for r in self._records if r.success)
        return {
            "total_calls": len(self._records),
            "total_successes": successes,
            "success_rate": successes / len(self._records) if self._records else 0,
            "total_elapsed_s": sum(r.elapsed_s for r in self._records),
        }


# Singleton for the current session
_session_metrics = MetricsCollector()


def get_session_metrics() -> MetricsCollector:
    """Get the current session's metrics collector."""
    return _session_metrics


def reset_session_metrics() -> MetricsCollector:
    """Reset and return a fresh metrics collector."""
    global _session_metrics
    _session_metrics = MetricsCollector()
    return _session_metrics


# ---------------------------------------------------------------------------
# Hook callbacks
# ---------------------------------------------------------------------------

# Dangerous tools that should be logged/gated
_DESTRUCTIVE_TOOLS = {"Bash", "Write", "apply_changes"}

# Tools that must not be called without prior analysis
_REQUIRES_CONTEXT = {"generate_code", "apply_changes"}


async def _pre_tool_use_validator(
    input_data: Dict[str, Any],
    tool_use_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate tool inputs before execution."""
    tool_name = input_data.get("tool_input", {}).get("name", "")

    # Log destructive operations
    if tool_name in _DESTRUCTIVE_TOOLS:
        logger.info("Destructive tool call: %s (id=%s)", tool_name, tool_use_id)

    return {}


async def _post_tool_use_recorder(
    input_data: Dict[str, Any],
    tool_use_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Record tool call metrics after execution."""
    tool_name = input_data.get("tool_input", {}).get("name", "unknown")
    # We don't have precise timing here, so record a placeholder
    _session_metrics.record_tool_call(
        tool_name=tool_name,
        elapsed_s=0.0,  # Agent SDK doesn't expose elapsed time in hooks
        success=True,  # PostToolUse only fires on success
    )
    return {}


async def _post_tool_use_failure_recorder(
    input_data: Dict[str, Any],
    tool_use_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Record failed tool calls."""
    tool_name = input_data.get("tool_input", {}).get("name", "unknown")
    _session_metrics.record_tool_call(
        tool_name=tool_name,
        elapsed_s=0.0,
        success=False,
    )
    return {}


async def _stop_hook(
    input_data: Dict[str, Any],
    tool_use_id: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Final hook when session ends — log summary metrics."""
    summary = _session_metrics.get_summary()
    logger.info("Session complete: %s", summary)
    return {}


def create_hooks(
    enable_validation: bool = True,
    enable_metrics: bool = True,
) -> Dict[str, List[Any]]:
    """Create the hook configuration dict for ClaudeAgentOptions.

    Returns a dict compatible with the Agent SDK hooks parameter.
    """
    hooks: Dict[str, list] = {}

    if enable_validation:
        hooks["PreToolUse"] = [
            {"matcher": ".*", "hooks": [_pre_tool_use_validator]},
        ]

    if enable_metrics:
        hooks.setdefault("PostToolUse", []).append(
            {"matcher": ".*", "hooks": [_post_tool_use_recorder]},
        )
        hooks.setdefault("PostToolUseFailure", []).append(
            {"matcher": ".*", "hooks": [_post_tool_use_failure_recorder]},
        )

    hooks["Stop"] = [
        {"matcher": ".*", "hooks": [_stop_hook]},
    ]

    return hooks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_agent_sdk_hooks.py -v`
Expected: PASS (all 7 tests)

- [ ] **Step 5: Commit**

```bash
git add core/agent_sdk/hooks.py tests/test_agent_sdk_hooks.py
git commit -m "feat: add hooks for quality gates, metrics, and feedback"
```

---

## Task 6: Main Controller

**Files:**
- Create: `core/agent_sdk/controller.py`
- Test: `tests/test_agent_sdk_controller.py`

This is the orchestration heart — it ties everything together.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_sdk_controller.py
"""Tests for Agent SDK meta-controller."""
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path


class TestControllerInit(unittest.TestCase):
    """Test controller initialization."""

    def test_create_controller(self):
        from core.agent_sdk.controller import AuraController
        from core.agent_sdk.config import AgentSDKConfig

        config = AgentSDKConfig()
        controller = AuraController(
            config=config,
            project_root=Path("/tmp/test"),
        )
        self.assertIsNotNone(controller)
        self.assertEqual(controller.project_root, Path("/tmp/test"))

    def test_controller_builds_options(self):
        from core.agent_sdk.controller import AuraController
        from core.agent_sdk.config import AgentSDKConfig

        config = AgentSDKConfig(model="claude-sonnet-4-6", max_turns=20)
        controller = AuraController(config=config, project_root=Path("/tmp/test"))
        options = controller._build_options(goal="Test goal")
        self.assertEqual(options.model, "claude-sonnet-4-6")
        self.assertEqual(options.max_turns, 20)

    def test_controller_builds_system_prompt(self):
        from core.agent_sdk.controller import AuraController
        from core.agent_sdk.config import AgentSDKConfig

        config = AgentSDKConfig()
        controller = AuraController(config=config, project_root=Path("/tmp/test"))
        prompt = controller._build_prompt(goal="Fix login bug")
        self.assertIn("Fix login bug", prompt)
        self.assertIn("AURA", prompt)

    def test_controller_registers_subagents(self):
        from core.agent_sdk.controller import AuraController
        from core.agent_sdk.config import AgentSDKConfig

        config = AgentSDKConfig(enable_subagents=True)
        controller = AuraController(config=config, project_root=Path("/tmp/test"))
        agents = controller._build_subagent_defs()
        self.assertIn("planning-agent", agents)
        self.assertIn("implementation-agent", agents)

    def test_controller_no_subagents_when_disabled(self):
        from core.agent_sdk.controller import AuraController
        from core.agent_sdk.config import AgentSDKConfig

        config = AgentSDKConfig(enable_subagents=False)
        controller = AuraController(config=config, project_root=Path("/tmp/test"))
        agents = controller._build_subagent_defs()
        self.assertEqual(agents, {})


class TestControllerMCPServer(unittest.TestCase):
    """Test that the controller creates an MCP server with AURA tools."""

    def test_mcp_server_created(self):
        from core.agent_sdk.controller import AuraController
        from core.agent_sdk.config import AgentSDKConfig

        config = AgentSDKConfig()
        controller = AuraController(config=config, project_root=Path("/tmp/test"))
        server = controller._build_mcp_server()
        self.assertIsNotNone(server)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agent_sdk_controller.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# core/agent_sdk/controller.py
"""AURA Meta-Controller — Agent SDK-powered closed-loop orchestrator.

Replaces rigid phase-sequenced orchestration with Claude-as-brain.
Claude dynamically selects tools, skills, agents, and workflows
based on goal context.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular deps and startup cost
_sdk_available: Optional[bool] = None


def _check_sdk() -> bool:
    """Check if claude-agent-sdk is installed."""
    global _sdk_available
    if _sdk_available is None:
        try:
            import claude_agent_sdk  # noqa: F401
            _sdk_available = True
        except ImportError:
            _sdk_available = False
    return _sdk_available


class AuraController:
    """Main entry point for Agent SDK-powered goal execution.

    Usage:
        controller = AuraController(config=config, project_root=root)
        result = await controller.run("Fix the authentication bug")
    """

    def __init__(
        self,
        config: Any,
        project_root: Path,
        brain: Any = None,
        model_adapter: Any = None,
        goal_queue: Any = None,
        goal_archive: Any = None,
    ) -> None:
        from core.agent_sdk.config import AgentSDKConfig

        self.config: AgentSDKConfig = config
        self.project_root = project_root
        self._brain = brain
        self._model_adapter = model_adapter
        self._goal_queue = goal_queue
        self._goal_archive = goal_archive

    def _build_prompt(self, goal: str) -> str:
        """Build the full prompt for the Agent SDK session."""
        from core.agent_sdk.context_builder import ContextBuilder

        builder = ContextBuilder(
            project_root=self.project_root,
            brain=self._brain,
        )
        context = builder.build(goal=goal)
        return builder.build_system_prompt(
            goal=goal,
            goal_type=context["goal_type"],
            context=context,
        )

    def _build_subagent_defs(self) -> Dict[str, Any]:
        """Build subagent definitions if enabled."""
        if not self.config.enable_subagents:
            return {}

        from core.agent_sdk.subagent_definitions import get_subagent_definitions

        if not _check_sdk():
            # Return raw defs — caller will need to wrap them
            return {
                name: defn
                for name, defn in get_subagent_definitions().items()
            }

        from claude_agent_sdk import AgentDefinition

        return {
            name: AgentDefinition(
                description=defn.description,
                prompt=defn.prompt,
                tools=defn.tools,
            )
            for name, defn in get_subagent_definitions().items()
        }

    def _build_mcp_server(self) -> Any:
        """Create an MCP server with all AURA tools registered."""
        from core.agent_sdk.tool_registry import create_aura_tools

        tools = create_aura_tools(
            project_root=self.project_root,
            brain=self._brain,
            model_adapter=self._model_adapter,
            goal_queue=self._goal_queue,
            goal_archive=self._goal_archive,
            config=self.config,
        )

        if not _check_sdk():
            # Return the raw tools list for testing without SDK
            return tools

        from claude_agent_sdk import tool, create_sdk_mcp_server

        sdk_tools = []
        for aura_tool in tools:
            @tool(aura_tool.name, aura_tool.description, aura_tool.input_schema.get("properties", {}))
            async def _handler(args, _t=aura_tool):
                result = _t.handler(args)
                import json
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps(result, default=str),
                    }]
                }
            sdk_tools.append(_handler)

        return create_sdk_mcp_server("aura-tools", tools=sdk_tools)

    def _build_hooks(self) -> Dict[str, Any]:
        """Build hooks configuration if enabled."""
        if not self.config.enable_hooks:
            return {}

        from core.agent_sdk.hooks import create_hooks, reset_session_metrics

        reset_session_metrics()
        return create_hooks()

    def _build_options(self, goal: str) -> Any:
        """Build ClaudeAgentOptions for a session."""
        system_prompt = self._build_prompt(goal)
        mcp_server = self._build_mcp_server()
        subagents = self._build_subagent_defs()
        hooks = self._build_hooks()

        if not _check_sdk():
            # Return a mock-like namespace for testing
            class _MockOptions:
                pass
            opts = _MockOptions()
            opts.model = self.config.model
            opts.max_turns = self.config.max_turns
            opts.system_prompt = system_prompt
            opts.agents = subagents
            return opts

        from claude_agent_sdk import ClaudeAgentOptions

        mcp_servers = {"aura": mcp_server}
        allowed = list(self.config.allowed_tools)

        return ClaudeAgentOptions(
            cwd=str(self.project_root),
            model=self.config.model,
            max_turns=self.config.max_turns,
            max_budget_usd=self.config.max_budget_usd,
            permission_mode=self.config.permission_mode,
            allowed_tools=allowed,
            system_prompt=system_prompt,
            mcp_servers=mcp_servers,
            agents=subagents,
            hooks=hooks,
            thinking=self.config.thinking_config,
        )

    async def run(self, goal: str) -> Dict[str, Any]:
        """Execute a goal using the Agent SDK meta-controller.

        Returns a dict with:
            - result: The final text output
            - session_id: For resumption
            - metrics: Tool call metrics from the session
        """
        if not _check_sdk():
            raise RuntimeError(
                "claude-agent-sdk not installed. "
                "Install with: pip install claude-agent-sdk"
            )

        from claude_agent_sdk import query, ResultMessage, SystemMessage
        from core.agent_sdk.hooks import get_session_metrics

        options = self._build_options(goal)
        session_id = None
        result_text = ""

        prompt = (
            f"Execute this development goal:\n\n{goal}\n\n"
            "Start by calling analyze_goal to understand the context, "
            "then proceed with the appropriate workflow."
        )

        async for message in query(prompt=prompt, options=options):
            if isinstance(message, SystemMessage) and message.subtype == "init":
                session_id = message.data.get("session_id")
            elif isinstance(message, ResultMessage):
                result_text = message.result

        metrics = get_session_metrics().get_summary()

        return {
            "result": result_text,
            "session_id": session_id,
            "metrics": metrics,
        }

    async def run_with_client(self, goal: str) -> Dict[str, Any]:
        """Execute using ClaudeSDKClient for full lifecycle control.

        Use this when you need to interrupt, resume, or manage MCP servers
        at runtime.
        """
        if not _check_sdk():
            raise RuntimeError("claude-agent-sdk not installed.")

        from claude_agent_sdk import (
            ClaudeSDKClient, AssistantMessage, TextBlock, ResultMessage,
        )
        from core.agent_sdk.hooks import get_session_metrics

        options = self._build_options(goal)
        result_text = ""

        prompt = (
            f"Execute this development goal:\n\n{goal}\n\n"
            "Start by calling analyze_goal to understand the context, "
            "then proceed with the appropriate workflow."
        )

        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            result_text += block.text
                elif isinstance(message, ResultMessage):
                    result_text = message.result

        metrics = get_session_metrics().get_summary()
        return {"result": result_text, "metrics": metrics}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_agent_sdk_controller.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add core/agent_sdk/controller.py tests/test_agent_sdk_controller.py
git commit -m "feat: add Agent SDK meta-controller with Claude-as-brain orchestration"
```

---

## Task 7: CLI Integration

**Files:**
- Create: `core/agent_sdk/cli_integration.py`
- Modify: `aura_cli/cli_main.py` (add `agent-run` command)
- Modify: `aura_cli/options.py` (add `agent-run` spec)
- Test: `tests/test_agent_sdk_cli_integration.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_sdk_cli_integration.py
"""Tests for Agent SDK CLI integration."""
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path


class TestCLIIntegration(unittest.TestCase):
    """Test CLI command wiring for agent-run."""

    def test_build_controller_from_args(self):
        from core.agent_sdk.cli_integration import build_controller_from_args

        args = MagicMock()
        args.model = None
        args.max_turns = None
        args.max_budget = None
        args.permission_mode = None
        args.project_root = "/tmp/test"

        controller = build_controller_from_args(args)
        self.assertIsNotNone(controller)

    def test_build_controller_with_overrides(self):
        from core.agent_sdk.cli_integration import build_controller_from_args

        args = MagicMock()
        args.model = "claude-opus-4-6"
        args.max_turns = 50
        args.max_budget = 10.0
        args.permission_mode = "bypassPermissions"
        args.project_root = "/tmp/test"

        controller = build_controller_from_args(args)
        self.assertEqual(controller.config.model, "claude-opus-4-6")
        self.assertEqual(controller.config.max_turns, 50)
        self.assertEqual(controller.config.max_budget_usd, 10.0)

    def test_format_result(self):
        from core.agent_sdk.cli_integration import format_result

        result = {
            "result": "Successfully fixed the bug.",
            "session_id": "abc-123",
            "metrics": {"total_calls": 5, "success_rate": 1.0},
        }
        output = format_result(result)
        self.assertIn("Successfully fixed", output)
        self.assertIn("abc-123", output)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agent_sdk_cli_integration.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# core/agent_sdk/cli_integration.py
"""Wire the Agent SDK meta-controller into AURA CLI commands."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def build_controller_from_args(args: Any) -> Any:
    """Build an AuraController from parsed CLI args."""
    from core.agent_sdk.config import AgentSDKConfig
    from core.agent_sdk.controller import AuraController

    # Load base config from aura.config.json if available
    config = AgentSDKConfig()
    try:
        config_path = Path("aura.config.json")
        if config_path.exists():
            with open(config_path) as f:
                aura_config = json.load(f)
            config = AgentSDKConfig.from_aura_config(aura_config)
    except Exception:
        pass  # Fall back to defaults

    # Apply CLI overrides
    if getattr(args, "model", None):
        config.model = args.model
    if getattr(args, "max_turns", None):
        config.max_turns = args.max_turns
    if getattr(args, "max_budget", None):
        config.max_budget_usd = args.max_budget
    if getattr(args, "permission_mode", None):
        config.permission_mode = args.permission_mode

    # Apply env overrides last
    config.apply_env_overrides()

    project_root = Path(getattr(args, "project_root", ".")).resolve()

    # Try to load brain and model adapter from existing AURA infra
    brain = None
    model_adapter = None
    try:
        from memory.brain import Brain
        brain = Brain()
    except Exception:
        pass
    try:
        from core.model_adapter import ModelAdapter
        model_adapter = ModelAdapter()
    except Exception:
        pass

    return AuraController(
        config=config,
        project_root=project_root,
        brain=brain,
        model_adapter=model_adapter,
    )


def format_result(result: Dict[str, Any]) -> str:
    """Format controller result for CLI output."""
    parts = []

    if result.get("result"):
        parts.append(result["result"])

    if result.get("session_id"):
        parts.append(f"\n--- Session: {result['session_id']} ---")

    if result.get("metrics"):
        m = result["metrics"]
        parts.append(
            f"Metrics: {m.get('total_calls', 0)} tool calls, "
            f"{m.get('success_rate', 0):.0%} success rate"
        )

    return "\n".join(parts)


async def handle_agent_run(args: Any) -> int:
    """CLI handler for 'agent-run' command."""
    goal = getattr(args, "goal", None)
    if not goal:
        print("Error: --goal is required")
        return 1

    controller = build_controller_from_args(args)

    try:
        result = await controller.run(goal)
        print(format_result(result))
        return 0
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1
    except Exception as exc:
        logger.exception("Agent run failed")
        print(f"Error: {exc}")
        return 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_agent_sdk_cli_integration.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Update __init__.py exports**

```python
# Update core/agent_sdk/__init__.py to export the controller
```

Add these exports to `core/agent_sdk/__init__.py`:
```python
from core.agent_sdk.config import AgentSDKConfig
from core.agent_sdk.controller import AuraController
from core.agent_sdk.cli_integration import build_controller_from_args, handle_agent_run

__all__ = [
    "AgentSDKConfig",
    "AuraController",
    "build_controller_from_args",
    "handle_agent_run",
]
```

- [ ] **Step 6: Commit**

```bash
git add core/agent_sdk/cli_integration.py core/agent_sdk/__init__.py tests/test_agent_sdk_cli_integration.py
git commit -m "feat: add CLI integration for agent-run command"
```

---

## Task 8: Wire into CLI Dispatch

**Files:**
- Modify: `aura_cli/options.py` (add agent-run command spec)
- Modify: `aura_cli/cli_main.py` (add dispatch entry)

- [ ] **Step 1: Read current CLI options structure**

Run: `python3 -m pytest tests/test_cli_main_dispatch.py -v --co` (collect only, see existing tests)

- [ ] **Step 2: Add agent-run command spec to options.py**

Add the new command spec to the existing command definitions in `aura_cli/options.py`. The exact insertion point depends on the existing structure — add it alongside other command specs:

```python
# In the command definitions section of aura_cli/options.py
# Add a new CommandSpec for agent-run:
{
    "name": "agent-run",
    "help": "Run a goal using the Agent SDK meta-controller (Claude-as-brain orchestration)",
    "args": [
        {"name": "--goal", "required": True, "help": "Development goal to execute"},
        {"name": "--model", "default": None, "help": "Model override (e.g., claude-opus-4-6)"},
        {"name": "--max-turns", "type": int, "default": None, "help": "Maximum agent turns"},
        {"name": "--max-budget", "type": float, "default": None, "help": "Maximum budget in USD"},
        {"name": "--permission-mode", "default": None, "choices": ["default", "acceptEdits", "bypassPermissions", "plan"]},
        {"name": "--project-root", "default": ".", "help": "Project root directory"},
    ],
}
```

- [ ] **Step 3: Add dispatch entry to cli_main.py**

In `aura_cli/cli_main.py`, add `"agent-run"` to the `COMMAND_DISPATCH_REGISTRY`:

```python
# In COMMAND_DISPATCH_REGISTRY dict:
"agent-run": _handle_agent_run,
```

And add the handler function:

```python
async def _handle_agent_run(args, **kwargs):
    """Dispatch to Agent SDK meta-controller."""
    from core.agent_sdk.cli_integration import handle_agent_run
    import asyncio
    return asyncio.run(handle_agent_run(args))
```

- [ ] **Step 4: Regenerate CLI docs**

Run: `python3 scripts/generate_cli_reference.py`

- [ ] **Step 5: Run CLI snapshot tests**

Run: `python3 -m pytest tests/test_cli_help_snapshots.py tests/test_cli_main_dispatch.py -v`
Expected: May need snapshot updates if output format changed

- [ ] **Step 6: Update snapshots if needed**

If snapshot tests fail due to intentional changes, update the affected files in `tests/snapshots/`.

- [ ] **Step 7: Commit**

```bash
git add aura_cli/options.py aura_cli/cli_main.py docs/CLI_REFERENCE.md tests/snapshots/
git commit -m "feat: wire agent-run command into CLI dispatch"
```

---

## Task 9: Add claude-agent-sdk Dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the dependency**

Add `claude-agent-sdk` to `requirements.txt`:

```
claude-agent-sdk>=0.1.0
```

- [ ] **Step 2: Verify install works**

Run: `pip install claude-agent-sdk` (or `pip install -e .` if using editable mode)

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add claude-agent-sdk dependency"
```

---

## Task 10: Integration Test

**Files:**
- Create: `tests/integration/test_agent_sdk_integration.py`

- [ ] **Step 1: Write integration test (SDK not required)**

```python
# tests/integration/test_agent_sdk_integration.py
"""Integration tests for Agent SDK meta-controller.

These tests verify the full assembly pipeline without requiring
the claude-agent-sdk to be installed — they test everything up to
the actual SDK call.
"""
import unittest
from pathlib import Path
from unittest.mock import MagicMock


class TestFullAssembly(unittest.TestCase):
    """Test that all components wire together correctly."""

    def test_controller_assembles_all_components(self):
        """Controller should build options with tools, subagents, and hooks."""
        from core.agent_sdk.config import AgentSDKConfig
        from core.agent_sdk.controller import AuraController

        config = AgentSDKConfig(
            model="claude-sonnet-4-6",
            max_turns=10,
            enable_subagents=True,
            enable_hooks=True,
        )
        controller = AuraController(
            config=config,
            project_root=Path("/tmp/test"),
        )

        # Should build without errors
        prompt = controller._build_prompt("Fix the auth bug")
        self.assertIn("Fix the auth bug", prompt)
        self.assertIn("bug_fix", prompt)

        subagents = controller._build_subagent_defs()
        self.assertGreater(len(subagents), 0)

        mcp_server = controller._build_mcp_server()
        self.assertIsNotNone(mcp_server)

    def test_tool_registry_handles_all_tools(self):
        """All registered tools should be callable with minimal args."""
        from core.agent_sdk.tool_registry import create_aura_tools

        tools = create_aura_tools(project_root=Path("/tmp/test"))

        # analyze_goal should work without any external deps
        analyze = next(t for t in tools if t.name == "analyze_goal")
        result = analyze.handler({"goal": "Add OAuth support"})
        self.assertEqual(result["goal_type"], "feature")
        self.assertIn("recommended_skills", result)

    def test_context_builder_end_to_end(self):
        """Context builder should produce a complete system prompt."""
        from core.agent_sdk.context_builder import ContextBuilder

        builder = ContextBuilder(project_root=Path("/tmp/test"))
        ctx = builder.build(goal="Refactor the database layer")
        prompt = builder.build_system_prompt(
            goal="Refactor the database layer",
            goal_type=ctx["goal_type"],
            context=ctx,
        )

        self.assertIn("Refactor the database layer", prompt)
        self.assertIn("refactor", prompt)
        self.assertIn("Recommended Skills", prompt)

    def test_metrics_survive_session(self):
        """Metrics should accumulate across tool calls."""
        from core.agent_sdk.hooks import MetricsCollector

        mc = MetricsCollector()
        mc.record_tool_call("analyze_goal", 0.1, True)
        mc.record_tool_call("create_plan", 2.5, True)
        mc.record_tool_call("generate_code", 5.0, True)
        mc.record_tool_call("verify_changes", 3.0, False)

        summary = mc.get_summary()
        self.assertEqual(summary["total_calls"], 4)
        self.assertEqual(summary["total_successes"], 3)
        self.assertAlmostEqual(summary["success_rate"], 0.75)

        stats = mc.get_stats()
        self.assertEqual(stats["tool_calls"]["analyze_goal"]["count"], 1)
        self.assertEqual(stats["tool_calls"]["verify_changes"]["success_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run integration test**

Run: `python3 -m pytest tests/integration/test_agent_sdk_integration.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 3: Run full test suite**

Run: `python3 -m pytest tests/test_agent_sdk_*.py tests/integration/test_agent_sdk_integration.py -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_agent_sdk_integration.py
git commit -m "test: add integration tests for Agent SDK meta-controller"
```

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│                    CLI: agent-run                        │
│  python3 main.py agent-run --goal "Fix auth bug"        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              AuraController.run(goal)                    │
│                                                         │
│  1. ContextBuilder → classify, gather hints, skills     │
│  2. Build system prompt with tool/agent inventory       │
│  3. Create SDK MCP server with 15 AURA tools            │
│  4. Register 4 subagents                                │
│  5. Configure hooks for metrics + quality gates         │
│  6. query(prompt, options) → Claude makes decisions     │
└──────────────────────┬──────────────────────────────────┘
                       │
            Claude decides what to do
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│ Custom Tools │ │ Subagents│ │ Built-in SDK │
│ (via MCP)    │ │          │ │ Tools        │
│              │ │ planning │ │              │
│ analyze_goal │ │ implement│ │ Read/Edit    │
│ create_plan  │ │ verify   │ │ Bash/Glob    │
│ generate_code│ │ research │ │ Grep/Write   │
│ verify_changes│ │         │ │              │
│ dispatch_skills│ │        │ │              │
│ search_memory│ │          │ │              │
│ invoke_mcp   │ │          │ │              │
│ run_workflow │ │          │ │              │
│ ...15 total  │ │          │ │              │
└──────┬───────┘ └────┬─────┘ └──────────────┘
       │              │
       ▼              ▼
┌─────────────────────────────────────────────────────────┐
│              AURA Infrastructure                         │
│                                                         │
│  agents/     → PlannerAgent, CriticAgent, CoderAgent... │
│  core/       → orchestrator, file_tools, goal_queue...  │
│  tools/      → 7 MCP servers (8001-8025)                │
│  memory/     → Brain (SQLite+NetworkX), decision log    │
│  agents/skills/ → 30+ static analysis skills            │
└─────────────────────────────────────────────────────────┘
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Claude-as-brain, not Claude-as-executor** | Claude decides *what* to do; AURA agents *do* it. Leverages existing tested infrastructure. |
| **Custom MCP tools, not direct agent calls** | Agent SDK's tool protocol gives Claude structured I/O with schemas. Tools are discoverable and self-documenting. |
| **Subagents for parallel work** | Complex goals decompose naturally into planning + implementation + verification that can run concurrently. |
| **Hooks for quality gates** | Pre/post hooks enforce safety policies without cluttering the main prompt. Metrics feed the reflection loop. |
| **Lazy imports everywhere** | Avoids circular deps and 2s+ startup cost from heavy imports (NetworkX, TextBlob). |
| **Graceful degradation** | Every tool handler catches exceptions and returns structured error dicts — Claude can adapt. |
| **Config hierarchy: defaults → aura.config.json → env vars → CLI args** | Same pattern as existing AURA config; easy to override at any level. |
