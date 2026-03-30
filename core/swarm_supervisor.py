"""Integration helpers for binding the hierarchical workflow into AURA runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from agents.hierarchical_coordinator import HierarchicalCoordinatorAgent
from agents.sdlc_debugger import SDLCDebuggerAgent
from core.swarm_models import AgentRole, SupervisorConfig
from memory.learning_loop import LessonStore


def register_swarm_agents(registry: Any) -> None:
    """Register coordinator and debugger with a best-effort adapter for registry variants."""
    _register(registry, "hierarchical_coordinator", HierarchicalCoordinatorAgent)
    _register(registry, "sdlc_debugger", SDLCDebuggerAgent)


def install_swarm_runtime(
    orchestrator: Any,
    registry: Any,
    config: SupervisorConfig | None = None,
    lessons_root: str | Path = ".aura_forge/memory",
) -> HierarchicalCoordinatorAgent:
    """Attach a run_hierarchical_story coroutine to the existing orchestrator instance."""
    active_config = config or SupervisorConfig()
    lesson_store = LessonStore(root_dir=lessons_root)
    coordinator = HierarchicalCoordinatorAgent(config=active_config, lesson_store=lesson_store)
    register_swarm_agents(registry)

    async def run_hierarchical_story(story_id: str, story_text: str, workers: Mapping[AgentRole, Any]) -> Any:
        return await coordinator.execute(story_id=story_id, story_text=story_text, workers=workers)

    setattr(orchestrator, "run_hierarchical_story", run_hierarchical_story)
    setattr(orchestrator, "swarm_supervisor", coordinator)
    return coordinator


def _register(registry: Any, name: str, factory: Callable[..., Any]) -> None:
    for method_name in ("register_factory", "register", "add_agent", "register_agent"):
        method = getattr(registry, method_name, None)
        if method is None:
            continue
        try:
            method(name, factory)
        except TypeError:
            try:
                method(name=name, factory=factory)
            except TypeError:
                continue
        return
