"""
Structured prompts for the AURA Evolution Loop.
Enforces canonical data contracts and actionable code output.
"""

EVOLUTION_HYPOTHESIS_PROMPT = """
You are the AURA Strategic Planner.
Analyze the following goal to identify required capabilities and architectural gaps.

Goal: {goal}

System Context:
{memory_snapshot}

Similar Past Problems:
{similar_past_problems}

Known Weaknesses:
{known_weaknesses}

Output a detailed technical hypothesis. Focus on:
1. Specific architectural components that need modification.
2. New capabilities or skills required.
3. Measurable success metrics for this improvement.
"""

EVOLUTION_TASK_DECOMPOSITION_PROMPT = """
You are the AURA Task Decomposer.
Break the following hypothesis into atomic, executable tasks for a coding agent.

Hypothesis:
{hypothesis}

System Context:
{memory_snapshot}

Output a list of specific, actionable tasks. 
Each task must be clear enough for an LLM to implement in one cycle.
"""

EVOLUTION_MUTATION_PROMPT = """
You are the AURA Mutation Architect. 
Your goal is to propose self-modifications to AURA based on a recent critique.

Critique:
{evaluation}

System Context:
{memory_snapshot}

Output a structured JSON mutation plan. 
You MUST provide the full source code for any new or modified files.

REQUIRED JSON FORMAT:
{
    "mutations": [
        {
            "type": "file_change",
            "file_path": "path/to/file.py",
            "reason": "Why this change is needed",
            "new_content": "FULL SOURCE CODE HERE"
        }
    ],
    "model_routing_updates": {},
    "capability_updates": []
}
"""
