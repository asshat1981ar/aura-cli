from __future__ import annotations
import random
from typing import Any, Dict, List, Optional
from pathlib import Path

from core.logging_utils import log_json
from agents.skills.structural_analyzer import StructuralAnalyzerSkill
from core.autonomous_discovery import AutonomousDiscovery

class ContextualGoalGenerator:
    """
    Generates impactful and contextually adaptive goals for AURA's RSI loop.
    
    It combines architectural hotspots (StructuralAnalyzer) with latent codebase
    signals (AutonomousDiscovery) to propose specific improvement targets.
    """

    DEFAULT_GOAL = "evolve and improve the AURA system via recursive self-improvement"

    def __init__(self, project_root: str | Path, brain=None, model=None):
        self.project_root = Path(project_root)
        self.brain = brain
        self.model = model
        self.analyzer = StructuralAnalyzerSkill(brain=brain, model=model)
        # AutonomousDiscovery needs a goal_queue and memory_store which we might not have here,
        # but we can run its scan_repo method directly if we instantiate it lightly.
        self.discovery = AutonomousDiscovery(None, None, project_root=str(project_root))

    def generate_impactful_goal(self) -> str:
        """
        Analyzes the codebase and returns the most impactful goal found.
        Falls back to DEFAULT_GOAL if no specific issues are detected.
        """
        try:
            # 1. Get architectural hotspots
            analysis = self.analyzer.run({"project_root": str(self.project_root)})
            hotspots = analysis.get("hotspots", [])
            cycles = analysis.get("circular_dependencies", [])

            # 2. Get discovery signals (TODOs, missing tests, etc.)
            discovery_report = self.discovery.run_scan()
            signals = discovery_report.get("items", [])

            # 3. Prioritize and select
            
            # Priority 1: Circular Dependencies
            if cycles:
                # Pick a cycle and suggest breaking it
                cycle = random.choice(cycles)
                target_modules = " -> ".join(cycle[:3]) + ("..." if len(cycle) > 3 else "")
                return f"Refactor and decouple circular dependency: {target_modules}"

            # Priority 2: Critical Hotspots (High Complexity + High Centrality)
            critical_hotspots = [h for h in hotspots if h.get("risk_level") == "CRITICAL"]
            if critical_hotspots:
                target = random.choice(critical_hotspots)
                return f"Refactor and optimize critical architectural hotspot: {target['file']} (Complexity: {target['max_complexity']})"

            # Priority 3: High Risk Hotspots
            high_risk_hotspots = [h for h in hotspots if h.get("risk_level") == "HIGH"]
            if high_risk_hotspots:
                target = random.choice(high_risk_hotspots)
                return f"Improve maintainability and reduce complexity of high-risk file: {target['file']}"

            # Priority 4: Missing Tests for central files
            bottlenecks = analysis.get("bottlenecks", [])
            missing_tests = [s for s in signals if s.get("type") == "missing_tests"]
            if missing_tests and bottlenecks:
                bottleneck_files = {b["file"] for b in bottlenecks[:5]}
                important_missing = [s for s in missing_tests if s.get("file") in bottleneck_files]
                if important_missing:
                    target = random.choice(important_missing)
                    return f"Implement missing regression tests for central module: {target['file']}"

            # Priority 5: High-priority TODOs or FIXMEs
            high_todo = [s for s in signals if s.get("priority") == "high" and s.get("type") == "todo_fixme"]
            if high_todo:
                target = random.choice(high_todo)
                return f"Resolve high-priority technical debt: {target['goal']}"

            # Priority 6: General Hotspots
            if hotspots:
                target = random.choice(hotspots)
                return f"Analyze and refactor architectural hotspot: {target['file']}"

        except Exception as e:
            log_json("ERROR", "goal_generation_failed", details={"error": str(e)})

        return self.DEFAULT_GOAL
