#!/usr/bin/env python3
"""
Autonomous RSI Multi-Cycle Run.
Uses the established EvolutionLoop to drive continuous self-improvement.
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Ensure project root is in path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from aura_cli.cli_main import default_agents
from core.evolution_loop import EvolutionLoop
from core.recursive_improvement import RecursiveImprovementService
from core.model_adapter import ModelAdapter
from core.vector_store import VectorStore
from memory.brain import Brain
from core.git_tools import GitTools
from agents.mutator import MutatorAgent

def main():
    max_cycles = 10
    goal = "evolve and improve the AURA system via recursive self-improvement"
    
    print(f">>> Starting {max_cycles} cycles of autonomous evolution...")
    
    brain = Brain()
    model = ModelAdapter()
    agents = default_agents(brain, model)
    
    # Extract raw agents from adapters
    _coder = getattr(agents.get("act"), "agent", agents.get("act"))
    _critic = getattr(agents.get("critique"), "agent", agents.get("critique"))
    _planner = getattr(agents.get("plan"), "agent", agents.get("plan"))
    
    git = GitTools(repo_path=str(project_root))
    mutator = MutatorAgent(project_root)
    vec = VectorStore(model, brain)
    ri_service = RecursiveImprovementService()
    
    evo = EvolutionLoop(
        planner=_planner,
        coder=_coder,
        critic=_critic,
        brain=brain,
        vector_store=vec,
        git_tools=git,
        mutator=mutator,
        improvement_service=ri_service
    )
    
    for i in range(1, max_cycles + 1):
        print(f"\n--- Starting Cycle {i}/{max_cycles} ---")
        try:
            result = evo.run(goal)
            print(f"--- Cycle {i} complete ---")
            # Trigger meta-improvement check
            evo.on_cycle_complete({"goal": goal, "status": "success"})
        except Exception as e:
            print(f"!!! Cycle {i} failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n>>> RSI Evolution run finished.")

if __name__ == "__main__":
    main()
