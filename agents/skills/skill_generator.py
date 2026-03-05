"""Skill: autonomously generate and register new AURA skills.

Analyzes recurring task patterns or explicit requests for new capabilities,
generates the Python implementation for a new SkillBase subclass,
persists it to agents/skills/, and registers it in the skill registry.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json

class SkillGeneratorSkill(SkillBase):
    """
    Skill for autonomous skill creation and registration.
    """

    name = "skill_generator"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        capability = input_data.get("capability")
        description = input_data.get("description", "")
        project_root = Path(input_data.get("project_root", "."))
        
        if not capability:
            return {"error": "No capability specified for skill generation"}

        if not self.model:
            return {"error": "SkillGeneratorSkill requires a model for generation"}

        log_json("INFO", "skill_generator_start", details={"capability": capability})

        # 1. Generate skill code
        skill_code = self._generate_skill_code(capability, description)
        if not skill_code:
            return {"error": "Failed to generate skill code"}

        # 2. Extract skill name and class name
        skill_name_match = re.search(r'name = "(.+?)"', skill_code)
        class_name_match = re.search(r'class (.+?)\(SkillBase\):', skill_code)
        
        if not skill_name_match or not class_name_match:
            return {"error": "Generated code is missing name or class definition"}
            
        skill_name = skill_name_match.group(1)
        class_name = class_name_match.group(1)
        file_name = f"{skill_name}.py"
        file_path = project_root / "agents" / "skills" / file_name

        # 3. Write skill file
        try:
            file_path.write_text(skill_code, encoding="utf-8")
            log_json("INFO", "skill_generator_file_written", details={"path": str(file_path)})
        except Exception as e:
            return {"error": f"Failed to write skill file: {e}"}

        # 4. Register skill
        registration_error = self._register_skill(project_root, skill_name, class_name, file_name)
        if registration_error:
            return {"error": f"Skill written but registration failed: {registration_error}"}

        # 5. Update dispatcher
        dispatcher_error = self._update_dispatcher(project_root, skill_name, capability)
        if dispatcher_error:
             log_json("WARN", "skill_generator_dispatcher_update_failed", details={"error": dispatcher_error})

        log_json("INFO", "skill_generator_complete", details={"skill": skill_name, "class": class_name})
        
        return {
            "status": "success",
            "skill_name": skill_name,
            "class_name": class_name,
            "file_path": str(file_path)
        }

    def _generate_skill_code(self, capability: str, description: str) -> str:
        prompt = f"""
You are the AURA Skill Generator. Your task is to write a new Python skill for the AURA system.

Capability requested: {capability}
Description: {description}

The skill must:
1. Inherit from `SkillBase` (from `agents.skills.base`).
2. Have a unique `name` attribute (snake_case).
3. Implement the `_run(self, input_data: Dict[str, Any]) -> Dict[str, Any]` method.
4. Use `log_json` for observability.
5. Be self-contained and robust.

Example structure:
```python
from __future__ import annotations
from typing import Any, Dict
from agents.skills.base import SkillBase
from core.logging_utils import log_json

class NewCapabilitySkill(SkillBase):
    name = "new_capability"
    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation here
        return {{"status": "complete"}}
```

Respond ONLY with the Python code for the skill, inside a markdown code block.
"""
        response = self.model.respond(prompt)
        match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _register_skill(self, project_root: Path, skill_name: str, class_name: str, file_name: str) -> Optional[str]:
        registry_path = project_root / "agents" / "skills" / "registry.py"
        if not registry_path.exists():
            return "Registry file not found"
            
        try:
            content = registry_path.read_text(encoding="utf-8")
            
            # 1. Add import
            import_line = f"    from agents.skills.{skill_name} import {class_name}"
            if import_line not in content:
                # Insert before the first SkillBase return
                content = content.replace("    from agents.skills.base import SkillBase", 
                                          f"    from agents.skills.base import SkillBase\n{import_line}")
            
            # 2. Add to dictionary
            entry = f'        "{skill_name}": {class_name}(brain=brain, model=model),'
            if entry not in content:
                # Insert before the closing brace of the dict
                content = content.replace("    }", f"{entry}\n    }}")
            
            registry_path.write_text(content, encoding="utf-8")
            return None
        except Exception as e:
            return str(e)

    def _update_dispatcher(self, project_root: Path, skill_name: str, capability: str) -> Optional[str]:
        dispatcher_path = project_root / "core" / "skill_dispatcher.py"
        if not dispatcher_path.exists():
            return "Dispatcher file not found"
            
        try:
            content = dispatcher_path.read_text(encoding="utf-8")
            # For simplicity, add to "default" list for now, or "feature" if it sounds like one
            if '"default": [' in content:
                content = content.replace('"default": [', f'"default": [\n        "{skill_name}",')
            
            dispatcher_path.write_text(content, encoding="utf-8")
            return None
        except Exception as e:
            return str(e)

