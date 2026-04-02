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

    def _get_memory_hints(self, goal: str) -> List[str]:
        """Retrieve relevant memory hints from brain.

        Brain.recall_with_budget(max_tokens) returns List[str] of
        recent memories. We filter by goal keywords for relevance.
        """
        if self._brain is None:
            return []
        try:
            memories = self._brain.recall_with_budget(max_tokens=2000)
            # Simple keyword filtering for relevance
            goal_words = set(goal.lower().split())
            relevant = [
                m for m in memories
                if any(w in m.lower() for w in goal_words)
            ]
            return relevant or memories[:5]
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
            if context.get("failure_patterns"):
                patterns = "\n".join(f"- {p}" for p in context["failure_patterns"])
                parts.append(f"### Failure Patterns\nRecent failures for this goal type:\n{patterns}")
            if context.get("skill_weights"):
                sorted_skills = sorted(
                    context["skill_weights"].items(), key=lambda x: x[1], reverse=True
                )
                weights_str = ", ".join(f"{name} ({w:.1f})" for name, w in sorted_skills)
                parts.append(f"### Skill Weights\n{weights_str}")
            if context.get("workflow_info"):
                parts.append(f"### Workflow\n{context['workflow_info']}")
            if context.get("model_tier"):
                parts.append(f"### Model Tier\n{context['model_tier']}")
            context_section = "\n\n".join(parts)

        return _SYSTEM_PROMPT_TEMPLATE.format(
            goal=goal,
            goal_type=goal_type,
            context_section=context_section,
        )
