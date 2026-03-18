"""Skill: autonomously generate and register new AURA Agents.

Analyzes requests for new workflow agents, generates the Python implementation
for an Agent subclass in the agents/ directory, and registers it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json

class AgentGeneratorSkill(SkillBase):
    """
    Skill for autonomous Agent creation and registration.
    """

    name = "agent_generator"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        agent_purpose = input_data.get("agent_purpose")
        description = input_data.get("description", "")
        project_root = Path(input_data.get("project_root", "."))
        
        if not agent_purpose:
            return {"error": "No agent_purpose specified for Agent generation"}

        if not self.model:
            return {"error": "AgentGeneratorSkill requires a model for generation"}

        log_json("INFO", "agent_generator_start", details={"agent_purpose": agent_purpose})

        # 1. Generate Agent code
        agent_code = self._generate_agent_code(agent_purpose, description)
        if not agent_code:
            return {"error": "Failed to generate Agent code"}

        # 2. Extract class name
        class_name_match = re.search(r'class (.+?)\(Agent\):', agent_code)
        
        if not class_name_match:
            return {"error": "Generated code is missing Agent class definition"}
            
        class_name = class_name_match.group(1)
        
        # Determine file name by converting camel case to snake case
        file_name_base = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
        if not file_name_base.endswith("_agent"):
            file_name_base += "_agent"
            
        file_name = f"{file_name_base}.py"
        file_path = project_root / "agents" / file_name

        # 3. Write agent file
        try:
            file_path.write_text(agent_code, encoding="utf-8")
            log_json("INFO", "agent_generator_file_written", details={"path": str(file_path)})
        except Exception as e:
            return {"error": f"Failed to write Agent file: {e}"}

        # 4. Register agent in registry (if needed, otherwise log)
        registration_error = self._register_agent(project_root, file_name_base, class_name)
        if registration_error:
            return {"error": f"Agent written but registration failed: {registration_error}"}
        
        log_json("INFO", "agent_generator_complete", details={"class_name": class_name})
        
        return {
            "status": "success",
            "class_name": class_name,
            "file_path": str(file_path)
        }

    def _generate_agent_code(self, agent_purpose: str, description: str) -> str:
        prompt = f"""
You are the AURA Agent Generator. Your task is to write a new Python Agent for the AURA system.

Agent Purpose: {agent_purpose}
Description: {description}

The agent must:
1. Inherit from `Agent` (from `agents.base`).
2. Have a clear class name (e.g., `MyNewAgent`).
3. Implement the `run(self, input_data: Dict) -> Dict` method.
4. Use `log_json` for observability.
5. Be self-contained.

Example structure:
```python
from typing import Dict
from agents.base import Agent
from core.logging_utils import log_json

class MyNewAgent(Agent):
    name = "my_new_agent"
    
    def __init__(self, brain=None, model=None):
        self.brain = brain
        self.model = model

    def run(self, input_data: Dict) -> Dict:
        log_json("INFO", f"{{self.name}}_started")
        # Implementation here
        return {{"status": "complete"}}
```

Respond ONLY with the Python code for the Agent, inside a markdown code block.
"""
        response = self.model.respond(prompt)
        match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _register_agent(self, project_root: Path, module_name: str, class_name: str) -> Optional[str]:
        registry_path = project_root / "agents" / "registry.py"
        if not registry_path.exists():
            return "Registry file not found"
            
        try:
            content = registry_path.read_text(encoding="utf-8")
            
            # 1. Add import
            import_line = f"from agents.{module_name} import {class_name}"
            if import_line not in content:
                # Insert alongside other imports
                if "from agents.coder import CoderAgent" in content:
                    content = content.replace("from agents.coder import CoderAgent", 
                                              f"from agents.coder import CoderAgent\n{import_line}")
            
            registry_path.write_text(content, encoding="utf-8")
            return None
        except Exception as e:
            return str(e)

