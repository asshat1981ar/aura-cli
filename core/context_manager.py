"""
Advanced Semantic Context Manager (ASCM).

Curates a "Context Bundle" for AURA agents by leveraging semantic search
(VectorStore) and relational data (ContextGraph).  It intelligently
allocates token budgets to ensure the most relevant information is
prioritised.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.config_manager import config
from core.logging_utils import log_json
from core.memory_types import RetrievalQuery, ContextBundle, SearchHit

class ContextManager:
    """
    ASCM v2: Manages the assembly and prioritization of agent context.
    Enforces deterministic token budgets and rich provenance tags.
    """

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
        
        # Load from unified config if available
        sem_mem = config.get("semantic_memory", {}) or {}
        self.max_tokens = sem_mem.get("budget_tokens", max_tokens)
        
        # Default budget allocation percentages
        self.budgets = {
            "goal": 0.1,
            "relevant_snippets": 0.5,
            "related_insights": 0.2,
            "memory": 0.1,
            "files": 0.1
        }
        # Override with config if present
        if "budgets" in sem_mem:
            self.budgets.update(sem_mem["budgets"])

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars / token)."""
        if not text: return 0
        return max(1, len(text) // 4)

    def get_context_bundle(
        self,
        goal: str,
        goal_type: str = "default",
        recent_memory: List[str] = None,
        file_list: List[str] = None
    ) -> Dict[str, Any]:
        """Assemble a prioritized context bundle for the given goal."""
        log_json("INFO", "ascm_assembling_context", details={"goal": goal[:80], "type": goal_type})

        # 1. Retrieve Semantic Snippets (Greedy Candidates)
        snippet_budget = int(self.max_tokens * self.budgets["relevant_snippets"])
        snippets = self._get_semantic_snippets(goal, snippet_budget)

        # 2. Retrieve Graph Insights
        insights = self._get_graph_insights(goal, goal_type)

        # 3. Assemble Bundle with Deterministic Budgeting
        used_tokens = 0
        budget_report = {}
        
        # A. Goal (Mandatory, First)
        goal_cost = self._estimate_tokens(goal)
        used_tokens += goal_cost
        budget_report["goal"] = goal_cost

        # B. Snippets (Primary Context)
        final_snippets = []
        snippet_tokens = 0
        for snip in snippets:
            cost = self._estimate_tokens(snip["content"])
            if snippet_tokens + cost <= snippet_budget:
                final_snippets.append(snip)
                snippet_tokens += cost
        used_tokens += snippet_tokens
        budget_report["snippets"] = snippet_tokens

        # C. Insights (Refined Context)
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

        # D. Session Memory (Temporal Context)
        final_memory = []
        memory_budget = int(self.max_tokens * self.budgets["memory"])
        memory_tokens = 0
        for mem in reversed(recent_memory or []):
            cost = self._estimate_tokens(mem)
            if memory_tokens + cost <= memory_budget:
                final_memory.insert(0, mem)
                memory_tokens += cost
            else:
                break 
        used_tokens += memory_tokens
        budget_report["memory"] = memory_tokens

        # E. Files (Structural Context)
        final_files = []
        files_budget = int(self.max_tokens * self.budgets["files"])
        files_tokens = 0
        for f in (file_list or []):
            cost = self._estimate_tokens(f) + 1
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

    def _get_semantic_snippets(self, goal: str, token_budget: int) -> List[Dict[str, Any]]:
        """Retrieve code snippets semantically related to the goal."""
        if not self.vs:
            return []
        
        # Over-fetch candidates then prune by score/budget
        sem_mem = config.get("semantic_memory", {}) or {}
        top_k = sem_mem.get("top_k", 15)
        min_score = sem_mem.get("min_score", 0.65)
        
        query = RetrievalQuery(
            query_text=goal,
            k=top_k,
            min_score=min_score,
            budget_tokens=token_budget
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
        if hasattr(self.cg, "best_skills_for_goal_type"):
            skills = self.cg.best_skills_for_goal_type(goal_type, limit=2)
            if skills:
                insights.append(f"Recommended skills: {', '.join(skills)}")

        # 2. Known weaknesses for this goal type
        if hasattr(self.cg, "weaknesses_for_goal_type"):
            weaknesses = self.cg.weaknesses_for_goal_type(goal_type, limit=2)
            if weaknesses:
                insights.append(f"Avoid these pitfalls: {', '.join(weaknesses)}")

        return insights

    def format_as_prompt(self, bundle: Dict[str, Any]) -> str:
        """
        R2: Provenance-tagged prompt formatting.
        Converts context bundle into a structured LLM prompt.
        """
        lines = [f"GOAL: {bundle['goal']}", f"TYPE: {bundle['goal_type']}", ""]
        
        if bundle.get("snippets"):
            lines.append("### SEMANTIC CONTEXT (PROVENANCE-TAGGED) ###")
            for snip in bundle["snippets"]:
                ref = snip.get("source") or "unknown"
                score = snip.get("score", 0.0)
                lines.append(f"--- [Source: {ref} | Confidence: {score:.2f}] ---")
                lines.append(snip["content"])
                lines.append("-" * 40)
            lines.append("")

        if bundle.get("related_insights"):
            lines.append("### GRAPH-DRIVEN INSIGHTS ###")
            for ins in bundle["related_insights"]:
                lines.append(f"💡 {ins}")
            lines.append("")

        if bundle.get("memory"):
            lines.append("### RECENT SESSION MEMORY ###")
            for m in bundle["memory"]:
                lines.append(f"⏱️ {m}")
            lines.append("")

        if bundle.get("files"):
            lines.append("### PROJECT STRUCTURE ###")
            lines.append(", ".join(bundle["files"]))
            
        return "\n".join(lines)
