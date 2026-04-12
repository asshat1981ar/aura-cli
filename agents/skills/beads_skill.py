"""Skill: interact with the Beads issue tracker (bd CLI).

Allows AURA to find ready tasks, claim them, and close them upon completion.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.logging_utils import log_json


class BeadsSkill(SkillBase):
    """
    Skill to wrap the 'bd' CLI for autonomous task management.

    Supported commands (via 'cmd' argument):
      - ready: List unblocked tasks
      - show: Show task details (requires 'id')
      - update: Update task status (requires 'id' and 'status' or '--claim')
      - close: Close task (requires 'id')
      - prime: Get AI context
      - sync: Persist changes (wraps 'bd dolt push/pull')
    """

    name = "beads_skill"

    def _normalize_payload(self, cmd: str, payload: Any) -> Dict[str, Any]:
        """Wrap BEADS CLI JSON into the dict-only skill contract."""
        if isinstance(payload, dict):
            return dict(payload)
        if isinstance(payload, list):
            key = "ready" if cmd == "ready" else "items"
            return {key: payload}
        return {"value": payload}

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        cmd: Optional[str] = input_data.get("cmd")
        bead_id: Optional[str] = input_data.get("id")
        args: List[str] = input_data.get("args", [])

        if not cmd:
            return {"error": "Provide 'cmd' (ready, show, update, close, prime, sync)"}

        # Base command
        full_cmd = ["bd", "--json", cmd]

        # Add ID if applicable
        if bead_id:
            full_cmd.append(bead_id)

        # Add extra args
        full_cmd.extend(args)

        try:
            log_json("INFO", "beads_skill_executing", details={"command": " ".join(full_cmd)})
            result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=30, check=False)

            # Handle non-zero exit codes (some commands might return JSON on error too)
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            try:
                data = self._normalize_payload(cmd, json.loads(stdout) if stdout else {})
                if result.returncode != 0:
                    data["returncode"] = result.returncode
                    data["stderr"] = stderr
                return data
            except json.JSONDecodeError:
                return {"returncode": result.returncode, "stdout": stdout, "stderr": stderr, "error": "Failed to parse JSON output from bd CLI"}

        except subprocess.TimeoutExpired:
            return {"error": "beads_cli_timeout", "command": " ".join(full_cmd)}
        except Exception as e:
            return {"error": str(e), "command": " ".join(full_cmd)}
