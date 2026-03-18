"""Skill: interact with the Beads issue tracker (bd CLI).

Allows AURA to find ready tasks, claim them, and close them upon completion.
"""
from __future__ import annotations

import json
import re
import subprocess
from typing import Any, Dict, List, Optional

from agents.skills.base import SkillBase
from core.beads_cli import resolve_beads_cli, uses_repo_local_beads_cli
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

    @staticmethod
    def _error_payload(command: list[str], error: str, *, stderr: str = "", returncode: int | None = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "error": error,
            "command": " ".join(command),
        }
        if stderr:
            payload["stderr"] = stderr
        if returncode is not None:
            payload["returncode"] = returncode
        return payload

    def _normalize_payload(self, cmd: str, payload: Any) -> Dict[str, Any]:
        """Wrap BEADS CLI JSON into the dict-only skill contract."""
        if isinstance(payload, dict):
            return dict(payload)
        if isinstance(payload, list):
            key = "ready" if cmd == "ready" else "items"
            return {key: payload}
        return {"value": payload}

    _ALLOWED_CMDS = {"ready", "show", "update", "close", "prime", "sync", "pull", "push", "dolt"}
    _BEAD_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-:.]+$")

    def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        cmd: Optional[str] = input_data.get("cmd")
        bead_id: Optional[str] = input_data.get("id")
        args: List[str] = input_data.get("args", [])

        if not cmd:
            return {"error": "Provide 'cmd' (ready, show, update, close, prime, sync)"}

        if cmd not in self._ALLOWED_CMDS:
            return {"error": f"Command '{cmd}' not allowed. Must be one of: {', '.join(sorted(self._ALLOWED_CMDS))}"}

        if bead_id and not self._BEAD_ID_PATTERN.match(bead_id):
            return {"error": f"Invalid bead ID format: '{bead_id}'"}

        # Validate args are simple strings without shell metacharacters
        for arg in args:
            if not isinstance(arg, str) or any(c in arg for c in ";|&$`\\'\"\n"):
                return {"error": f"Invalid argument: '{arg}'"}

        beads_cli = resolve_beads_cli()
        full_cmd = [beads_cli]
        if uses_repo_local_beads_cli(beads_cli):
            full_cmd.append("--no-daemon")
        full_cmd.extend([cmd, "--json"])

        # Add ID if applicable
        if bead_id:
            full_cmd.append(bead_id)

        # Add extra args
        full_cmd.extend(args)

        try:
            log_json("INFO", "beads_skill_executing", details={"command": " ".join(full_cmd)})
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False
            )
            
            # Handle non-zero exit codes (some commands might return JSON on error too)
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            
            try:
                data = self._normalize_payload(cmd, json.loads(stdout) if stdout else {})
                if result.returncode != 0:
                    if "beads_cli_unavailable" in stderr or "beads_cli_unavailable" in stdout:
                        return self._error_payload(full_cmd, "capability_unavailable", stderr=stderr or stdout, returncode=result.returncode)
                    data["returncode"] = result.returncode
                    data["stderr"] = stderr
                return data
            except json.JSONDecodeError:
                return self._error_payload(
                    full_cmd,
                    "Failed to parse JSON output from bd CLI",
                    stderr=stderr or stdout,
                    returncode=result.returncode,
                )

        except subprocess.TimeoutExpired:
            return self._error_payload(full_cmd, "beads_cli_timeout")
        except FileNotFoundError as exc:
            return self._error_payload(full_cmd, "capability_unavailable", stderr=str(exc))
        except Exception as e:
            return self._error_payload(full_cmd, str(e))
