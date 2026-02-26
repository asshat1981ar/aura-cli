"""
Advanced Semantic Context Manager (ASCM).

Curates a "Context Bundle" for AURA agents by leveraging semantic search
(VectorStore) and relational data (ContextGraph).  It intelligently
allocates token budgets to ensure the most relevant information is
prioritised.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logging_utils import log_json

class ContextManager:
    """Manages the assembly and prioritization of agent context."""

    def __init__(
        self,
        vector_store=None,
        context_graph=None,
        project_root: str = ".",
        max_tokens: int = 8000
    ):
        self.vs = vector_store
        self.cg = context_graph
        self.root = Path(project_root)
        self.max_tokens = max_tokens
        # Budget allocation percentages (total 1.0)
        self.budgets = {
            "goal": 0.1,
            "relevant_snippets": 0.5,
            "related_files": 0.2,
            "memory": 0.1,
            "file_list": 0.1
        }

    def get_context_bundle(
        self,
        goal: str,
        goal_type: str = "default",
        recent_memory: List[str] = None,
        file_list: List[str] = None
    ) -> Dict[str, Any]:
        """Assemble a prioritized context bundle for the given goal."""
        log_json("INFO", "ascm_assembling_context", details={"goal": goal[:80], "type": goal_type})

        bundle = {
            "goal": goal,
            "goal_type": goal_type,
            "snippets": self._get_semantic_snippets(goal),
            "related_insights": self._get_graph_insights(goal, goal_type),
            "memory": self._prioritize_memory(recent_memory or []),
            "files": self._truncate_file_list(file_list or [])
        }

        # Final assembly into a structured dict for the agent
        return bundle

    def _get_semantic_snippets(self, goal: str) -> List[Dict[str, str]]:
        """Retrieve code snippets semantically related to the goal."""
        if not self.vs:
            return []
        
        # Query vector store for top 3 similar contents
        results = self.vs.search(goal, k=3)
        snippets = []
        for res in results:
            # res is likely the content. We might need a way to know WHICH file it came from.
            # For now, we'll treat it as a blob.
            snippets.append({"content": res[:1000]}) # Cap each snippet
        return snippets

    def _get_graph_insights(self, goal: str, goal_type: str) -> List[str]:
        """Retrieve insights from the context graph."""
        insights = []
        if not self.cg:
            return insights

        # 1. Best skills for this goal type
        skills = self.cg.best_skills_for_goal_type(goal_type, limit=2)
        if skills:
            insights.append(f"High-value skills for this task: {', '.join(skills)}")

        # 2. Known weaknesses for this goal type
        weaknesses = self.cg.weaknesses_for_goal_type(goal_type, limit=2)
        if weaknesses:
            insights.append(f"Potential pitfalls to avoid: {', '.join(weaknesses)}")

        return insights

    def _prioritize_memory(self, memory: List[str]) -> List[str]:
        """Keep only the most relevant/recent memory entries."""
        # Current implementation: just keep last 5
        return memory[-5:]

    def _truncate_file_list(self, files: List[str]) -> List[str]:
        """Ensure file list doesn't consume too much of the budget."""
        # Current implementation: cap at 50
        return sorted(files)[:50]

    def format_as_prompt(self, bundle: Dict[str, Any]) -> str:
        """Helper to convert a bundle into a string suitable for an LLM prompt."""
        lines = [f"GOAL: {bundle['goal']}", f"TYPE: {bundle['goal_type']}", ""]
        
        if bundle["snippets"]:
            lines.append("### RELEVANT CODE SNIPPETS ###")
            for i, snip in enumerate(bundle["snippets"], 1):
                lines.append(f"Snippet {i}:")
                lines.append("```python")
                lines.append(snip["content"])
                lines.append("```")
            lines.append("")

        if bundle["related_insights"]:
            lines.append("### CONTEXTUAL INSIGHTS ###")
            for ins in bundle["related_insights"]:
                lines.append(f"- {ins}")
            lines.append("")

        if bundle["memory"]:
            lines.append("### RECENT SESSION HISTORY ###")
            for m in bundle["memory"]:
                lines.append(f"- {m}")
            lines.append("")

        if bundle["files"]:
            lines.append("### PROJECT STRUCTURE (TRUNCATED) ###")
            lines.append(", ".join(bundle["files"]))
            
        return "\n".join(lines)
