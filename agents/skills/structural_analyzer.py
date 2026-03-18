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

    def __init__(self, brain=None, model=None, context_graph: Optional[ContextGraph] = None):
        super().__init__(brain=brain, model=model)
        self.cg = context_graph or ContextGraph()
        # Initialize sub-skills with the same brain and model
        self.complexity_scorer = ComplexityScorerSkill(brain=brain, model=model)
        self.symbol_indexer = SymbolIndexerSkill(brain=brain, model=model)

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        project_root = input_data.get("project_root")
        if not project_root:
            return {"error": "Provide 'project_root'"}

        top_k = input_data.get("top_k", 10)

        # 1. Populate graph from symbol indexer
        # Try to use existing index if available to save time
        index_results = self.symbol_indexer.run({"project_root": project_root})
        import_graph = index_results.get("import_graph", {})
        
        if not import_graph:
            log_json("WARN", "structural_analyzer_no_imports_found")
        
        # Build mapping from module path to file path
        mod_to_file = {}
        for f in import_graph.keys():
            if f.endswith(".py"):
                mod_name = f[:-3].replace("/", ".").replace("\\", ".")
                mod_to_file[mod_name] = f
                if mod_name.endswith(".__init__"):
                    mod_to_file[mod_name[:-9]] = f

        for file_path, imports in import_graph.items():
            for imp in imports:
                resolved = mod_to_file.get(imp)
                if resolved:
                    self.cg.add_edge(file_path, resolved, "imports")
                else:
                    self.cg.add_edge(file_path, f"module:{imp}", "imports")

        # 2. Graph Analysis
        try:
            cycles = self.cg.find_circular_dependencies()
            bottlenecks = self.cg.find_bottleneck_files(limit=top_k * 2)
        except Exception as e:
            log_json("ERROR", "structural_analyzer_graph_analysis_failed", details={"error": str(e)})
            cycles = []
            bottlenecks = []

        # 3. Complexity Analysis
        comp_results = self.complexity_scorer.run({"project_root": project_root})
        detailed_comp = comp_results.get("file_results", {})

        # 4. Identify Hotspots (High Centrality + High Complexity)
        hotspots = []
        for b in bottlenecks:
            fname = b["file"]
            if fname.startswith("module:"):
                continue
                
            centrality = b["centrality_score"]
            funcs = detailed_comp.get(fname, [])
            max_cc = max([f.get("complexity", 0) for f in funcs], default=0)
            
            # Thresholds for hotspots
            if max_cc > 8 and centrality > 0.01:
                hotspots.append({
                    "file": fname,
                    "centrality": centrality,
                    "max_complexity": max_cc,
                    "risk_level": "CRITICAL" if max_cc > 20 or (max_cc > 15 and centrality > 0.1) else ("HIGH" if max_cc > 12 else "MEDIUM")
                })

        hotspots.sort(key=lambda x: (x["centrality"] * x["max_complexity"]), reverse=True)
        hotspots = hotspots[:top_k]

        # 5. Coverage Analysis (Optional)
        coverage_gaps = []
        if input_data.get("report_coverage"):
            try:
                from agents.skills.test_coverage_analyzer import TestCoverageAnalyzerSkill
                cov_analyzer = TestCoverageAnalyzerSkill(brain=self.brain, model=self.model)
                cov_results = cov_analyzer.run({"project_root": project_root})
                
                monitored_files = set([h["file"] for h in hotspots] + [b["file"] for b in bottlenecks])
                for f in cov_results.get("missing_files", []):
                    rel_f = f
                    if Path(f).is_absolute():
                        try:
                            rel_f = str(Path(f).relative_to(project_root))
                        except ValueError:
                            pass
                    
                    if rel_f in monitored_files:
                        coverage_gaps.append({
                            "file": rel_f,
                            "coverage_pct": 0.0,
                            "risk_priority": "HIGH"
                        })
            except Exception as e:
                log_json("WARN", "structural_analyzer_coverage_failed", details={"error": str(e)})

        summary = f"Detected {len(cycles)} circular dependencies and {len(hotspots)} hotspots."
        
        return {
            "circular_dependencies": cycles[:top_k],
            "bottlenecks": bottlenecks[:top_k],
            "hotspots": hotspots,
            "coverage_gaps": coverage_gaps[:top_k],
            "summary": summary
        }
