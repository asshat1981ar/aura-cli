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
        project_root_str = inputs.get("project_root", ".")
        project_root = Path(project_root_str)
        
        log_json("INFO", "evolution_skill_start", details={"goal": goal})
        
        if not self.brain or not self.model:
            return {"status": "error", "error": "EvolutionSkill requires both brain and model instances."}

        try:
            # Lazy import to avoid circular dependencies
            from core.evolution_loop import EvolutionLoop
            from agents.planner import PlannerAgent
            from core.git_tools import GitTools
            from agents.mutator import MutatorAgent
            
            planner = PlannerAgent(self.brain, self.model)
            git_tools = GitTools(repo_path=str(project_root))
            mutator = MutatorAgent(project_root)
            
            # Note: We don't have easy access to the orchestrator's act/critique agents here,
            # but EvolutionLoop can instantiate its own if needed, or we can pass None 
            # if it handles default creation. 
            # Looking at EvolutionLoop.__init__, it seems to require them.
            
            from agents.coder import CoderAgent
            from agents.critic import CriticAgent
            
            coder = CoderAgent(self.brain, self.model)
            critic = CriticAgent(self.brain, self.model)

            evo = EvolutionLoop(
                planner=planner,
                coder=coder,
                critic=critic,
                brain=self.brain,
                vector_store=getattr(self.brain, "vector_store", None),
                git_tools=git_tools,
                mutator=mutator
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
