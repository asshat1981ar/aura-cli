"""Task routing and load balancing for the orchestrator hub.

Extends the keyword-matching approach from core/capability_manager.py's
analyze_capability_needs() with agent capability vectors and round-robin
load balancing.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from orchestrator_hub.registry import AgentInfo, AgentRegistryHub


# Keyword → capability mapping for task routing
_TASK_KEYWORDS: Dict[str, List[str]] = {
    "python": ["python", "pytest", "pip", "flask", "django", "fastapi", "pydantic"],
    "typescript": ["typescript", "node", "npm", "react", "nextjs", "express", "deno"],
    "monitoring": ["monitor", "health", "metrics", "alert", "observability", "telemetry"],
    "notification": ["notify", "alert", "slack", "discord", "email", "pagerduty", "webhook"],
    "external_llm": ["llm", "gpt", "claude", "gemini", "openai", "anthropic", "model"],
    "code_generation": ["implement", "code", "generate", "write", "create", "build"],
    "planning": ["plan", "design", "architect", "strategy", "roadmap"],
    "testing": ["test", "verify", "validate", "check", "lint", "coverage"],
    "debugging": ["debug", "fix", "error", "bug", "crash", "exception"],
    "review": ["review", "critique", "analyze", "audit", "inspect"],
}


class TaskRouter:
    """Routes tasks to the best available agent based on capability matching."""

    def __init__(self, registry: AgentRegistryHub):
        self._registry = registry
        self._round_robin: Dict[str, int] = {}

    def route(self, task: dict) -> Optional[str]:
        """Route a task to the best agent.

        Args:
            task: Dict with at least "goal" or "task" key containing the
                natural-language description.

        Returns:
            Agent name, or None if no suitable agent found.
        """
        text = (task.get("goal") or task.get("task") or "").lower()
        if not text:
            return None

        # Score each capability based on keyword matches
        scores: Dict[str, int] = {}
        for capability, keywords in _TASK_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[capability] = score

        if not scores:
            return None

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Find an agent for the top-scoring capability
        for capability, _ in ranked:
            agents = self._registry.discover(capability)
            if agents:
                return self._pick_agent(capability, agents)

        return None

    def route_with_fallback(
        self, task: dict, exclude: Optional[List[str]] = None
    ) -> Optional[str]:
        """Route a task, excluding specific agents.

        Useful for retrying with a different agent after failure.
        """
        excluded = set(exclude or [])
        text = (task.get("goal") or task.get("task") or "").lower()
        if not text:
            return None

        scores: Dict[str, int] = {}
        for capability, keywords in _TASK_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[capability] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        for capability, _ in ranked:
            agents = [
                a for a in self._registry.discover(capability)
                if a.name not in excluded
            ]
            if agents:
                return self._pick_agent(capability, agents)

        return None

    def _pick_agent(self, capability: str, agents: List[AgentInfo]) -> str:
        """Round-robin selection among agents for a capability."""
        if not agents:
            return None
        idx = self._round_robin.get(capability, 0) % len(agents)
        self._round_robin[capability] = idx + 1
        return agents[idx].name

    def load_balance(self, agent_type: str) -> Optional[str]:
        """Round-robin across all agents of a given type."""
        agents = self._registry.discover_by_type(agent_type)
        if not agents:
            return None
        idx = self._round_robin.get(agent_type, 0) % len(agents)
        self._round_robin[agent_type] = idx + 1
        return agents[idx].name
