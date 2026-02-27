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
from dataclasses import asdict

from core.logging_utils import log_json
from core.memory_types import RetrievalQuery, ContextBundle, SearchHit

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
            "related_insights": 0.2,
            "memory": 0.1,
            "files": 0.1
        }

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars / token)."""
        return len(text) // 4

    def get_context_bundle(
        self,
        goal: str,
        goal_type: str = "default",
        recent_memory: List[str] = None,
        file_list: List[str] = None
    ) -> Dict[str, Any]:
        """Assemble a prioritized context bundle for the given goal."""
        log_json("INFO", "ascm_assembling_context", details={"goal": goal[:80], "type": goal_type})

        # Adaptive Strategy: Adjust budgets based on past context failures
        self._adjust_budgets(goal_type)

        recent_memory = recent_memory or []
        file_list = file_list or []

        # 1. Retrieve Semantic Snippets
        snippets = self._get_semantic_snippets(goal, self.max_tokens * self.budgets["relevant_snippets"])

        # 2. Retrieve Insights
        insights = self._get_graph_insights(goal, goal_type)

        # 3. Assemble Bundle with Budgeting
        used_tokens = 0
        budget_report = {}
        
        # Goal (mandatory)
        used_tokens += self._estimate_tokens(goal)
        budget_report["goal"] = self._estimate_tokens(goal)

        # Snippets
        final_snippets = []
        snippet_budget = int(self.max_tokens * self.budgets["relevant_snippets"])
        snippet_tokens = 0
        for snip in snippets:
            cost = self._estimate_tokens(snip["content"])
            if snippet_tokens + cost <= snippet_budget:
                final_snippets.append(snip)
                snippet_tokens += cost
        used_tokens += snippet_tokens
        budget_report["snippets"] = snippet_tokens

        # Insights
        final_insights = []
        insight_budget = int(self.max_tokens * self.budgets["related_insights"])
        insight_tokens = 0
        for ins in insights:
            cost = self._estimate_tokens(ins)
            if insight_tokens + cost <= insight_budget:
                final_insights.append(ins)
                insight_tokens += cost
        used_tokens += insight_tokens
        budget_report["insights"] = insight_tokens

        # Memory
        final_memory = []
        memory_budget = int(self.max_tokens * self.budgets["memory"])
        memory_tokens = 0
        # Prioritize recent memory (reverse order loop, keep order)
        for mem in reversed(recent_memory):
            cost = self._estimate_tokens(mem)
            if memory_tokens + cost <= memory_budget:
                final_memory.insert(0, mem)
                memory_tokens += cost
            else:
                break 
        used_tokens += memory_tokens
        budget_report["memory"] = memory_tokens

        # Files
        final_files = []
        files_budget = int(self.max_tokens * self.budgets["files"])
        files_tokens = 0
        for f in file_list:
            cost = self._estimate_tokens(f) + 1 # +1 for newline/separator
            if files_tokens + cost <= files_budget:
                final_files.append(f)
                files_tokens += cost
            else:
                break
        used_tokens += files_tokens
        budget_report["files"] = files_tokens
        
        budget_report["total_used"] = used_tokens
        budget_report["total_limit"] = self.max_tokens

        bundle = ContextBundle(
            goal=goal,
            goal_type=goal_type,
            snippets=final_snippets,
            related_insights=final_insights,
            memory=final_memory,
            files=final_files,
            budget_report=budget_report,
            trace={"vs_hits": len(snippets), "cg_hits": len(insights)}
        )

        return asdict(bundle)

    def _adjust_budgets(self, goal_type: str):
        """Boost context budget if we detect 'context_gap' weaknesses."""
        if not self.cg:
            return
            
        weaknesses = self.cg.weaknesses_for_goal_type(goal_type, limit=10)
        context_gaps = [w for w in weaknesses if "context_gap" in w]
        
        if context_gaps:
            log_json("INFO", "ascm_adaptive_boost", details={"reason": "context_gaps_detected", "count": len(context_gaps)})
            # Boost snippets and insights at the expense of raw file lists
            self.budgets["relevant_snippets"] = 0.6  # +10%
            self.budgets["related_insights"] = 0.25  # +5%
            self.budgets["files"] = 0.05             # -5%

    def _get_semantic_snippets(self, goal: str, token_budget: float) -> List[Dict[str, Any]]:
        """Retrieve code snippets semantically related to the goal."""
        if not self.vs:
            return []
        
        # Calculate k based on budget approximation (assuming 200 tokens per snippet avg)
        est_k = max(3, int(token_budget / 200))
        
        query = RetrievalQuery(
            query_text=goal,
            k=est_k,
            min_score=0.6,
            budget_tokens=int(token_budget)
        )
        
        hits = self.vs.search(query)
        
        snippets = []
        for hit in hits:
            snippets.append({
                "content": hit.content,
                "source": hit.source_ref,
                "score": hit.score,
                "explanation": hit.explanation
            })
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

    def format_as_prompt(self, bundle: Dict[str, Any]) -> str:
        """Helper to convert a bundle into a string suitable for an LLM prompt."""
        lines = [f"GOAL: {bundle['goal']}", f"TYPE: {bundle['goal_type']}", ""]
        
        if bundle["snippets"]:
            lines.append("### RELEVANT CODE SNIPPETS ###")
            for i, snip in enumerate(bundle["snippets"], 1):
                header = f"Snippet {i}"
                if snip.get("source"):
                    header += f" [Source: {snip['source']}]"
                if snip.get("score"):
                    header += f" [Score: {snip['score']:.2f}]"
                lines.append(header)
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
