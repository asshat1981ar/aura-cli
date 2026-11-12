"""Memory manager -- coordinates all memory modules and provides ReAct hooks.

The MemoryManager is the single entry point for the rest of the AURA system
to interact with long-term and short-term memory.  It owns instances of every
memory module and exposes:

* Unified search across all modules.
* Bulk export / import for backup and migration.
* ``pre_think_hook`` / ``post_observe_hook`` for ReAct loop integration.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

from core.logging_utils import log_json
from memory.episodic_memory import EpisodicMemory
from memory.memory_module import MemoryEntry, MemoryModule
from memory.procedural_memory import ProceduralMemory
from memory.semantic_memory import SemanticMemory
from memory.working_memory import WorkingMemory

_DEFAULT_DB_DIR = os.path.join(os.path.expanduser("~"), ".aura", "memory")

# Default token budget for memory injections in the pre-think hook.
_DEFAULT_TOKEN_BUDGET = 2000
# Approximate chars-per-token ratio (same heuristic used in brain.py).
_CHARS_PER_TOKEN = 4


class MemoryManager:
    """Coordinates all memory modules and provides ReAct hook integration.

    Args:
        config: Optional configuration dict.  Recognised keys:

            - ``db_dir`` (str): Root directory for SQLite databases.
            - ``working_max_entries`` (int): Ring-buffer size for working memory.
            - ``token_budget`` (int): Max tokens for memory injection in hooks.
            - ``summarize_fn``: Callback for working-memory eviction summaries.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        db_dir = cfg.get("db_dir", _DEFAULT_DB_DIR)

        self.working = WorkingMemory(
            max_entries=cfg.get("working_max_entries", 50),
            summarize_fn=cfg.get("summarize_fn"),
        )
        self.episodic = EpisodicMemory(
            db_path=os.path.join(db_dir, "episodic.db"),
        )
        self.semantic = SemanticMemory(
            db_path=os.path.join(db_dir, "semantic.db"),
        )
        self.procedural = ProceduralMemory(
            db_path=os.path.join(db_dir, "procedural.db"),
        )

        self._modules: dict[str, MemoryModule] = {
            "working": self.working,
            "episodic": self.episodic,
            "semantic": self.semantic,
            "procedural": self.procedural,
        }

        self._token_budget = cfg.get("token_budget", _DEFAULT_TOKEN_BUDGET)

        log_json(
            "INFO",
            "memory_manager_init",
            details={
                "db_dir": db_dir,
                "modules": list(self._modules.keys()),
                "token_budget": self._token_budget,
            },
        )

    # ------------------------------------------------------------------
    # Unified search
    # ------------------------------------------------------------------

    def search_all(
        self, query: str, top_k: int = 5
    ) -> dict[str, list[MemoryEntry]]:
        """Search across all modules and return results keyed by module name.

        Each module independently returns up to *top_k* results for *query*.
        """
        results: dict[str, list[MemoryEntry]] = {}
        for name, module in self._modules.items():
            try:
                results[name] = module.search(query, top_k=top_k)
            except Exception as exc:
                log_json(
                    "WARN",
                    "memory_manager_search_error",
                    details={"module": name, "error": str(exc)},
                )
                results[name] = []
        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, dict]:
        """Aggregate statistics from every module."""
        result: dict[str, dict] = {}
        for name, module in self._modules.items():
            try:
                result[name] = module.stats()
            except Exception as exc:
                log_json(
                    "WARN",
                    "memory_manager_stats_error",
                    details={"module": name, "error": str(exc)},
                )
                result[name] = {"error": str(exc)}
        return result

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export_all(self, path: str) -> None:
        """Export all memories to a JSON file at *path*.

        The file format is ``{"module_name": [serialised entries], ...}``.
        """
        data: dict[str, list[dict]] = {}
        for name, module in self._modules.items():
            try:
                entries = module.search("", top_k=10_000)
                data[name] = [
                    {
                        "id": e.id,
                        "content": e.content,
                        "metadata": e.metadata,
                        "timestamp": e.timestamp,
                    }
                    for e in entries
                ]
            except Exception as exc:
                log_json(
                    "WARN",
                    "memory_manager_export_error",
                    details={"module": name, "error": str(exc)},
                )
                data[name] = []

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        log_json(
            "INFO",
            "memory_manager_export",
            details={"path": path, "modules": list(data.keys())},
        )

    def import_all(self, path: str) -> None:
        """Import memories from a JSON file previously created by :meth:`export_all`."""
        if not os.path.exists(path):
            log_json(
                "WARN",
                "memory_manager_import_missing",
                details={"path": path},
            )
            return

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        for name, entries in data.items():
            module = self._modules.get(name)
            if module is None:
                log_json(
                    "WARN",
                    "memory_manager_import_unknown_module",
                    details={"module": name},
                )
                continue
            for entry in entries:
                try:
                    module.write(
                        content=entry.get("content", ""),
                        metadata=entry.get("metadata"),
                    )
                except Exception as exc:
                    log_json(
                        "WARN",
                        "memory_manager_import_entry_error",
                        details={
                            "module": name,
                            "entry_id": entry.get("id"),
                            "error": str(exc),
                        },
                    )

        log_json(
            "INFO",
            "memory_manager_import",
            details={"path": path},
        )

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def clear(self, module_name: Optional[str] = None) -> None:
        """Clear a specific module or all modules.

        Args:
            module_name: If provided, only that module is cleared.
                Otherwise every module is cleared.
        """
        if module_name:
            module = self._modules.get(module_name)
            if module is None:
                log_json(
                    "WARN",
                    "memory_manager_clear_unknown",
                    details={"module": module_name},
                )
                return
            module.clear()
            log_json(
                "INFO",
                "memory_manager_clear",
                details={"module": module_name},
            )
        else:
            for name, module in self._modules.items():
                try:
                    module.clear()
                except Exception as exc:
                    log_json(
                        "WARN",
                        "memory_manager_clear_error",
                        details={"module": name, "error": str(exc)},
                    )
            log_json("INFO", "memory_manager_clear_all")

    # ------------------------------------------------------------------
    # ReAct hook integration
    # ------------------------------------------------------------------

    def pre_think_hook(self, context: dict) -> dict:
        """Query all modules and inject relevant memories before the think step.

        The *context* dict is expected to have a ``task`` or ``thought`` key
        whose value is used as the search query.  The returned context will
        have an additional ``memory_context`` key containing relevant entries
        from each module, trimmed to fit the configured token budget.

        Args:
            context: The current ReAct context dict.

        Returns:
            The *context* dict augmented with ``memory_context``.
        """
        query = context.get("task") or context.get("thought") or ""
        if not query:
            return context

        budget_chars = self._token_budget * _CHARS_PER_TOKEN
        memory_context: dict[str, list[str]] = {}
        chars_used = 0

        # Query each module, allocating budget roughly evenly.
        per_module_budget = budget_chars // max(len(self._modules), 1)

        for name, module in self._modules.items():
            try:
                entries = module.read(str(query), top_k=5)
            except Exception as exc:
                log_json(
                    "WARN",
                    "memory_manager_pre_think_error",
                    details={"module": name, "error": str(exc)},
                )
                continue

            snippets: list[str] = []
            module_chars = 0
            for entry in entries:
                cost = len(entry.content)
                if module_chars + cost > per_module_budget:
                    break
                snippets.append(entry.content)
                module_chars += cost
            chars_used += module_chars
            if snippets:
                memory_context[name] = snippets

        context["memory_context"] = memory_context
        log_json(
            "INFO",
            "memory_manager_pre_think",
            details={
                "query_snippet": str(query)[:80],
                "modules_with_results": list(memory_context.keys()),
                "chars_injected": chars_used,
            },
        )
        return context

    def post_observe_hook(self, context: dict) -> dict:
        """Write observations to working memory after the observe step.

        Expects *context* to contain an ``observation`` key.  If the context
        also signals task completion (``task_complete`` is truthy), the
        observation is additionally written to episodic memory as a task
        trace.

        Args:
            context: The current ReAct context dict.

        Returns:
            The unmodified *context* dict (mutations are side-effects only).
        """
        observation = context.get("observation")
        if not observation:
            return context

        # Always store in working memory.
        self.working.write(
            content=str(observation),
            metadata={"source": "observe_hook"},
        )

        # On task completion, persist to episodic memory.
        if context.get("task_complete"):
            task_meta = {
                "task_id": context.get("task_id"),
                "task_type": context.get("task_type"),
                "outcome": context.get("outcome", "completed"),
            }
            self.episodic.write(
                content=str(observation),
                metadata=task_meta,
            )
            log_json(
                "INFO",
                "memory_manager_post_observe_episodic",
                details={"task_id": task_meta.get("task_id")},
            )

        log_json(
            "INFO",
            "memory_manager_post_observe",
            details={"observation_snippet": str(observation)[:80]},
        )
        return context
