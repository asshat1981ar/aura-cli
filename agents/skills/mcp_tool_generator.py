"""Skill: autonomously generate and register new AURA MCP tools.

Analyzes requests for new tools, generates the Python implementation 
for a FastMCP tool or adds it to an existing server, and registers it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json

class McpToolGeneratorSkill(SkillBase):
    """
    Skill for autonomous MCP tool creation and registration.
    """

    name = "mcp_tool_generator"

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        tool_purpose = input_data.get("tool_purpose")
        description = input_data.get("description", "")
        project_root = Path(input_data.get("project_root", "."))
        
        if not tool_purpose:
            return {"error": "No tool_purpose specified for MCP tool generation"}

        if not self.model:
            return {"error": "McpToolGeneratorSkill requires a model for generation"}

        log_json("INFO", "mcp_tool_generator_start", details={"tool_purpose": tool_purpose})

        # 1. Generate MCP Tool code
        tool_code = self._generate_mcp_tool_code(tool_purpose, description)
        if not tool_code:
            return {"error": "Failed to generate MCP tool code"}

        # 2. Extract tool name
        # We will assume a standalone FastMCP server script is generated, e.g. "my_new_tool_mcp.py"
        tool_name_match = re.search(r'name=["\'](.*?)["\']', tool_code)
        if not tool_name_match:
            tool_name_match = re.search(r'def\s+([a-zA-Z0-9_]+)\s*\(', tool_code)
            
        tool_name = tool_name_match.group(1).lower().replace(" ", "_") if tool_name_match else f"generated_tool_{hash(tool_code)}"
        
        file_name = f"{tool_name}_mcp.py"
        file_path = project_root / "tools" / file_name

        # 3. Write tool file
        try:
            file_path.write_text(tool_code, encoding="utf-8")
            log_json("INFO", "mcp_tool_generator_file_written", details={"path": str(file_path)})
        except Exception as e:
            return {"error": f"Failed to write MCP tool file: {e}"}

        # 4. Note on how to run it
        run_command = f"python3 {file_path}"
        
        log_json("INFO", "mcp_tool_generator_complete", details={"tool_name": tool_name})
        
        return {
            "status": "success",
            "tool_name": tool_name,
            "file_path": str(file_path),
            "run_command": run_command
        }

    def _generate_mcp_tool_code(self, tool_purpose: str, description: str) -> str:
        prompt = f"""
You are the AURA MCP Tool Generator. Your task is to write a new Python MCP Server (using FastMCP from mcp.server.fastmcp) for the AURA system.

Tool Purpose: {tool_purpose}
Description: {description}

The tool must:
1. Initialize a FastMCP server: `from mcp.server.fastmcp import FastMCP`
2. Define a clear `@mcp.tool()` function.
3. Be fully self-contained and run via `mcp.run()`.

Example structure:
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MyNewTool")

@mcp.tool()
def execute_my_tool(param1: str) -> str:
    # Tool description
    return f"Executed with {{param1}}"

if __name__ == "__main__":
    mcp.run()
```

Respond ONLY with the Python code for the MCP tool, inside a markdown code block.
"""
        response = self.model.respond(prompt)
        match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        return match.group(1).strip() if match else ""
