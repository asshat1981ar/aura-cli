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
        bottlenecks = self.cg.find_bottleneck_files(limit=20)

        # 3. Complexity Analysis
        comp_results = self.complexity_scorer.run({"project_root": project_root})
        detailed_comp = comp_results.get("file_results", {})

        # 4. Identify Hotspots (High Centrality + High Complexity)
        hotspots = []
        for b in bottlenecks:
            fname = b["file"]
            centrality = b["centrality_score"]
            
            # Find max complexity in this file
            funcs = detailed_comp.get(fname, [])
            max_cc = max([f["complexity"] for f in funcs], default=0)
            
            if max_cc > 8 and centrality > 0.03: # Heuristic thresholds
                hotspots.append({
                    "file": fname,
                    "centrality": centrality,
                    "max_complexity": max_cc,
                    "risk_level": "CRITICAL" if max_cc > 20 else ("HIGH" if max_cc > 12 else "MEDIUM")
                })

        # 5. Coverage Analysis (Optional)
        coverage_gaps = []
        if input_data.get("report_coverage"):
            from agents.skills.test_coverage_analyzer import TestCoverageAnalyzerSkill
            cov_analyzer = TestCoverageAnalyzerSkill()
            cov_results = cov_analyzer.run({"project_root": project_root})
            
            # Identify high-risk files with low/zero coverage
            # For now, focus on files identified as bottlenecks or hotspots
            monitored_files = set([h["file"] for h in hotspots] + [b["file"] for b in bottlenecks])
            
            # Missing files (0% coverage)
            for f in cov_results.get("missing_files", []):
                # Clean path if absolute
                rel_f = f
                if Path(f).is_absolute():
                    try:
                        rel_f = str(Path(f).relative_to(project_root))
                    except ValueError:
                        pass
                
                coverage_gaps.append({
                    "file": rel_f,
                    "coverage_pct": 0.0,
                    "risk_priority": "HIGH" if rel_f in monitored_files else "MEDIUM"
                })

        summary = f"Detected {len(cycles)} cycles and {len(hotspots)} architectural hotspots."
        
        res = {
            "circular_dependencies": cycles,
            "bottlenecks": bottlenecks,
            "hotspots": hotspots,
            "summary": summary
        }

        if input_data.get("report_coverage"):
            if coverage_gaps:
                res["summary"] += f" Found {len(coverage_gaps)} coverage gaps."
            res["coverage_gaps"] = coverage_gaps

        return res
