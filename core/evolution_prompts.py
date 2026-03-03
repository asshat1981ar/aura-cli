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

Output a list of mutation commands. 
Use the following commands ONLY:

ADD_FILE <path>
<full content of the file>

REPLACE_IN_FILE <path>
---OLD_CONTENT_START---
<exact code snippet to be replaced>
---OLD_CONTENT_END---
---NEW_CONTENT_START---
<new code snippet to replace the old one>
---NEW_CONTENT_END---

Example:
REPLACE_IN_FILE core/config_manager.py
---OLD_CONTENT_START---
    def get(self, key):
        return self._config.get(key)
---OLD_CONTENT_END---
---NEW_CONTENT_START---
    def get(self, key, default=None):
        return self._config.get(key, default)
---NEW_CONTENT_END---
"""
