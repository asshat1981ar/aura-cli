"""
Session initialization protocol inspired by openclaw-workspace patterns.

Provides structured session startup: agent config -> project context -> memory loading.
Each step is isolated and gracefully handles missing dependencies.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logging_utils import log_json


@dataclass
class SessionContext:
    """Immutable snapshot of everything needed to begin a session."""

    agent_config: Dict[str, Any] = field(default_factory=dict)
    project_context: Dict[str, Any] = field(default_factory=dict)
    recent_memories: List[Any] = field(default_factory=list)
    long_term_memories: List[Any] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    # -----------------------------------------------------------------
    # Prompt-ready serialisation
    # -----------------------------------------------------------------
    def to_prompt_context(self) -> str:
        """Render the session context into a prompt-friendly string.

        Sections are only included when they carry meaningful data so the
        prompt stays compact when some subsystems are unavailable.
        """
        sections: List[str] = []

        # --- Agent configuration ---
        if self.agent_config:
            parts = [f"  {k}: {v}" for k, v in self.agent_config.items()]
            sections.append("## Agent Configuration\n" + "\n".join(parts))

        # --- Project context ---
        if self.project_context:
            parts = [f"  {k}: {v}" for k, v in self.project_context.items()]
            sections.append("## Project Context\n" + "\n".join(parts))

        # --- Recent memories ---
        if self.recent_memories:
            items = [f"  - {m}" for m in self.recent_memories[:20]]
            sections.append("## Recent Memories\n" + "\n".join(items))

        # --- Long-term memories ---
        if self.long_term_memories:
            items = [f"  - {m}" for m in self.long_term_memories[:20]]
            sections.append("## Long-Term Memories\n" + "\n".join(items))

        if not sections:
            return "# Session Context\nNo context available."

        return "# Session Context\n\n" + "\n\n".join(sections)


class SessionInitializer:
    """Orchestrates the multi-step session startup sequence.

    Steps (all wrapped in try/except for graceful degradation):
      1. Load agent configuration from ``core.config_manager``
      2. Detect project context (pyproject.toml, .git, package.json)
      3. Load recent memories (SESSION tier + decision log)
      4. Load long-term memories (PROJECT tier + semantic/procedural modules)

    Any combination of optional dependencies may be ``None`` — the
    initializer will simply skip the corresponding step and log a
    warning.
    """

    def __init__(
        self,
        memory_manager: Any = None,
        memory_controller: Any = None,
        memory_store: Any = None,
        config: Any = None,
    ):
        self.memory_manager = memory_manager
        self.memory_controller = memory_controller
        self.memory_store = memory_store
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def initialize(self, task_hint: str = "") -> SessionContext:
        """Run the full initialisation sequence and return a SessionContext."""
        log_json("INFO", "session_init_start", details={"task_hint": task_hint})

        agent_config = self._load_agent_config()
        project_context = self._load_project_context()
        recent_memories = self._load_recent_memories()
        long_term_memories = self._load_long_term_memories()

        ctx = SessionContext(
            agent_config=agent_config,
            project_context=project_context,
            recent_memories=recent_memories,
            long_term_memories=long_term_memories,
        )

        log_json(
            "INFO",
            "session_init_complete",
            details={
                "agent_config_keys": len(agent_config),
                "project_context_keys": len(project_context),
                "recent_memories_count": len(recent_memories),
                "long_term_memories_count": len(long_term_memories),
            },
        )
        return ctx

    # ------------------------------------------------------------------
    # Step 1 — Agent configuration
    # ------------------------------------------------------------------
    def _load_agent_config(self) -> Dict[str, Any]:
        try:
            if self.config is not None:
                cfg = self.config
            else:
                from core.config_manager import config as _cfg

                cfg = _cfg

            result: Dict[str, Any] = {}
            for key in ("model_name", "dry_run", "max_cycles", "max_iterations"):
                val = cfg.get(key) if hasattr(cfg, "get") else None
                if val is not None:
                    result[key] = val

            log_json("INFO", "session_init_agent_config_loaded",
                     details={"keys": list(result.keys())})
            return result
        except Exception as exc:
            log_json("WARN", "session_init_agent_config_failed",
                     details={"error": str(exc)})
            return {}

    # ------------------------------------------------------------------
    # Step 2 — Project context
    # ------------------------------------------------------------------
    def _load_project_context(self) -> Dict[str, Any]:
        try:
            context: Dict[str, Any] = {}
            root = self._detect_project_root()
            if root:
                context["project_root"] = str(root)

            cwd = Path.cwd()
            pyproject = cwd / "pyproject.toml"
            if pyproject.exists():
                context["has_pyproject"] = True

            git_dir = cwd / ".git"
            if git_dir.exists():
                context["has_git"] = True

            package_json = cwd / "package.json"
            if package_json.exists():
                context["has_package_json"] = True

            log_json("INFO", "session_init_project_context_loaded",
                     details={"keys": list(context.keys())})
            return context
        except Exception as exc:
            log_json("WARN", "session_init_project_context_failed",
                     details={"error": str(exc)})
            return {}

    # ------------------------------------------------------------------
    # Step 3 — Recent memories
    # ------------------------------------------------------------------
    def _load_recent_memories(self) -> List[Any]:
        memories: List[Any] = []
        try:
            if self.memory_controller is not None:
                from memory.controller import MemoryTier

                session_entries = self.memory_controller.retrieve(
                    MemoryTier.SESSION, limit=20
                )
                memories.extend(session_entries)
                log_json("INFO", "session_init_session_memories_loaded",
                         details={"count": len(session_entries)})
        except Exception as exc:
            log_json("WARN", "session_init_session_memories_failed",
                     details={"error": str(exc)})

        try:
            if self.memory_store is not None:
                recent_decisions = self.memory_store.query(
                    "decision_log", limit=10
                )
                memories.extend(recent_decisions)
                log_json("INFO", "session_init_decision_log_loaded",
                         details={"count": len(recent_decisions)})
        except Exception as exc:
            log_json("WARN", "session_init_decision_log_failed",
                     details={"error": str(exc)})

        return memories

    # ------------------------------------------------------------------
    # Step 4 — Long-term memories
    # ------------------------------------------------------------------
    def _load_long_term_memories(self) -> List[Any]:
        memories: List[Any] = []
        try:
            if self.memory_controller is not None:
                from memory.controller import MemoryTier

                project_entries = self.memory_controller.retrieve(
                    MemoryTier.PROJECT, limit=20
                )
                memories.extend(project_entries)
                log_json("INFO", "session_init_project_memories_loaded",
                         details={"count": len(project_entries)})
        except Exception as exc:
            log_json("WARN", "session_init_project_memories_failed",
                     details={"error": str(exc)})

        try:
            if self.memory_manager is not None:
                if hasattr(self.memory_manager, "recall"):
                    semantic = self.memory_manager.recall("session_context", limit=10)
                    if semantic:
                        memories.extend(semantic)
                        log_json("INFO", "session_init_semantic_loaded",
                                 details={"count": len(semantic)})
                if hasattr(self.memory_manager, "get_procedures"):
                    procedural = self.memory_manager.get_procedures(limit=10)
                    if procedural:
                        memories.extend(procedural)
                        log_json("INFO", "session_init_procedural_loaded",
                                 details={"count": len(procedural)})
        except Exception as exc:
            log_json("WARN", "session_init_long_term_failed",
                     details={"error": str(exc)})

        return memories

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_project_root() -> Optional[Path]:
        """Walk up from cwd looking for common project markers."""
        markers = ("pyproject.toml", ".git", "package.json", "setup.py", "setup.cfg")
        current = Path.cwd()
        for _ in range(10):  # cap depth
            for marker in markers:
                if (current / marker).exists():
                    return current
            parent = current.parent
            if parent == current:
                break
            current = parent
        return None
