import json
from typing import Dict, Any, Optional

from core.logging_utils import log_json
from core.file_tools import _aura_safe_loads

ROUTER_PROMPT = """
You are AURA's Task Router. Your job is to map a high-level plan step to a specific skill execution.

Available Skills:
{skill_descriptions}

Current Context:
Goal: {goal}
Current Task: {task}

Output a JSON object with:
1. "skill": The name of the skill to use (or null if no skill matches).
2. "args": A dictionary of arguments for the skill.
3. "reasoning": A brief explanation of why this skill was chosen.

Rules:
- If the task requires thinking or planning (no tool), return "skill": null.
- If the task corresponds to a known skill, extract the arguments from the task description and context.
- Argument types must match the skill definition.
- Prefer specific skills (e.g., 'security_scanner') over generic ones if available.

Example:
Task: "Scan the auth module for vulnerabilities"
Output: {{
  "skill": "security_scanner",
  "args": {{ "path": "src/auth" }},
  "reasoning": "The user wants to find vulnerabilities in a specific path."
}}

Respond ONLY with the JSON object.
"""

class TaskRouter:
    """
    TaskRouter: Maps natural language plan steps to specific skill executions.
    """
    def __init__(self, model):
        self.model = model
        self.skill_registry = {}
        
    def _load_registry(self):
        if not self.skill_registry:
            from agents.skills.registry import all_skills
            self.skill_registry = all_skills()

    def route_task(self, goal: str, task: str) -> Dict[str, Any]:
        """
        Determines which skill to run for a given task.
        """
        self._load_registry()
        
        # 1. Simple heuristic checks (optimization)
        task_lower = task.lower()
        if "scan" in task_lower and "security" in task_lower:
             return {"skill": "security_scanner", "args": {"path": "."}, "reasoning": "Heuristic match"}
        
        if "lint" in task_lower:
             return {"skill": "linter_enforcer", "args": {"path": "."}, "reasoning": "Heuristic match"}

        # 2. LLM-based routing
        # Generate skill descriptions
        skills_desc = []
        for name, skill in self.skill_registry.items():
            doc = skill.__doc__.strip() if skill.__doc__ else "No description."
            # Grab first line only for brevity
            doc = doc.split('\n')[0]
            skills_desc.append(f"- {name}: {doc}")
        
        skills_text = "\n".join(skills_desc)
        
        # Truncate if too long (approx 4k chars safety)
        if len(skills_text) > 4000:
            skills_text = skills_text[:4000] + "... (truncated)"
            
        prompt = ROUTER_PROMPT.format(
            skill_descriptions=skills_text,
            goal=goal,
            task=task
        )
        
        try:
            response = self.model.respond(prompt)
            data = _aura_safe_loads(response, "task_router_response")
            
            if isinstance(data, dict):
                skill_name = data.get("skill")
                if skill_name:
                    if skill_name not in self.skill_registry:
                        log_json("WARN", "router_hallucinated_skill", {"skill": skill_name})
                        data["skill"] = None
                        data["reasoning"] = f"Hallucinated skill '{skill_name}' rejected."
                return data
            else:
                 return {"skill": None, "args": {}, "reasoning": "Invalid JSON response from router"}
                 
        except Exception as e:
            log_json("ERROR", "router_llm_failure", {"error": str(e)})
            return {"skill": None, "args": {}, "reasoning": f"Routing error: {e}"}
