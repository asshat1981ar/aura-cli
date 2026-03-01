"""
Structural Analyzer Skill — Detects architectural debt and hotspots.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path

from agents.skills.base import SkillBase
from core.context_graph import ContextGraph
from agents.skills.complexity_scorer import ComplexityScorerSkill

from agents.skills.symbol_indexer import SymbolIndexerSkill

class StructuralAnalyzerSkill(SkillBase):
    """
    Analyzes the ContextGraph for circular dependencies and bottlenecks,
    combining with complexity data to identify high-risk architectural hotspots.
    """
    name = "structural_analyzer"

    def __init__(self, context_graph: Optional[ContextGraph] = None):
        self.cg = context_graph or ContextGraph()
        self.complexity_scorer = ComplexityScorerSkill()
        self.symbol_indexer = SymbolIndexerSkill()

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        project_root = input_data.get("project_root")
        if not project_root:
            return {"error": "Provide 'project_root'"}

        # 1. Populate graph from symbol indexer if it looks empty or needs refresh
        index_results = self.symbol_indexer.run({"project_root": project_root})
        import_graph = index_results.get("import_graph", {})
        
        # Build mapping from module path to file path
        # core.logging_utils -> core/logging_utils.py
        mod_to_file = {}
        for f in import_graph.keys():
            if f.endswith(".py"):
                mod_name = f[:-3].replace("/", ".").replace("\\", ".")
                mod_to_file[mod_name] = f
                # Also handle __init__.py modules
                if mod_name.endswith(".__init__"):
                    mod_to_file[mod_name[:-9]] = f

        for file_path, imports in import_graph.items():
            for imp in imports:
                # Try resolving to project file
                resolved = mod_to_file.get(imp)
                if resolved:
                    self.cg.add_edge(file_path, resolved, "imports")
                else:
                    # Fallback for 3rd party or unresolved local
                    self.cg.add_edge(file_path, f"module:{imp}", "imports")

        # 2. Graph Analysis
        cycles = self.cg.find_circular_dependencies()
        bottlenecks = self.cg.find_bottleneck_files(limit=10)

        # 2. Complexity Analysis
        comp_results = self.complexity_scorer.run({"project_root": project_root})
        file_complexities = comp_results.get("file_avg_complexity", 0) # This is global avg
        
        # We need per-file complexity for hotspots
        # Extract from file_results if present
        detailed_comp = comp_results.get("file_results", {})

        # 3. Identify Hotspots (High Centrality + High Complexity)
        hotspots = []
        for b in bottlenecks:
            fname = b["file"]
            centrality = b["centrality_score"]
            
            # Find max complexity in this file
            funcs = detailed_comp.get(fname, [])
            max_cc = max([f["complexity"] for f in funcs], default=0)
            
            if max_cc > 10 and centrality > 0.05: # Heuristic thresholds
                hotspots.append({
                    "file": fname,
                    "centrality": centrality,
                    "max_complexity": max_cc,
                    "risk_level": "CRITICAL" if max_cc > 20 else "HIGH"
                })

        return {
            "circular_dependencies": cycles,
            "bottlenecks": bottlenecks,
            "hotspots": hotspots,
            "summary": f"Detected {len(cycles)} cycles and {len(hotspots)} architectural hotspots."
        }
