"""
EvolutionSkill — AURA's recursive self-improvement skill.
Wraps core/evolution_loop.py to allow LoopOrchestrator to trigger
evolutionary updates as a first-class skill.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json

class EvolutionSkill(SkillBase):
    """
    Skill wrapper for EvolutionLoop. 
    Enables AURA to hypothesize and mutate its own codebase.
    """
    name = "evolution_skill"

    def _run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run one cycle of the evolution loop.
        
        Inputs:
            goal (str): The evolutionary goal (e.g. "optimize model routing").
            project_root (str): Root path for mutation and git commits.
        """
        goal = inputs.get("goal", "evolve and improve the AURA system")
        project_root = Path(inputs.get("project_root", "."))
        
        log_json("INFO", "evolution_skill_start", details={"goal": goal})
        
        try:
            # Lazy import to avoid circular dependencies
            from core.evolution_loop import EvolutionLoop
            from core.orchestrator import LoopOrchestrator
            from aura_cli.cli_main import create_runtime
            
            # Use the global runtime or create a temporary one
            runtime = create_runtime(project_root)
            
            evo = EvolutionLoop(
                planner=runtime["planner"],
                coder=runtime["orchestrator"].agents.get("act"),
                critic=runtime["orchestrator"].agents.get("critique"),
                brain=runtime["brain"],
                vector_store=runtime["vector_store"],
                git_tools=runtime["git_tools"],
                mutator=runtime.get("mutator") or __import__("agents.mutator", fromlist=["MutatorAgent"]).MutatorAgent(project_root)
            )
            
            result = evo.run(goal)
            
            return {
                "status": "success",
                "goal": goal,
                "hypothesis": result.get("hypothesis"),
                "tasks": result.get("tasks"),
                "mutation": result.get("mutation")
            }
            
        except Exception as exc:
            log_json("ERROR", "evolution_skill_failed", details={"error": str(exc)})
            return {"status": "error", "error": str(exc)}
