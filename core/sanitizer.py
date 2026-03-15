import os
import re
from pathlib import Path
from typing import Any, List, Union
from core.exceptions import AuraError
from core.config_manager import config
from core.redaction import mask_secrets

class SecurityError(AuraError):
    """Raised when a security boundary is violated."""
    pass

# Hardcoded safe commands for AURA
BASE_ALLOWED_COMMANDS = {
    "python", "python3", "python3.10", "python3.11", "python3.12", "pytest", "git", "ls", "cat", "rm", "mkdir", "mv", "grep", "sed", "npx"
}

def get_allowed_commands() -> set:
    """Returns the effective set of allowed commands from config + base."""
    extra = config.get("security", {}).get("allowed_commands", [])
    return BASE_ALLOWED_COMMANDS.union(set(extra))

def sanitize_path(file_path: Union[str, Path], root_dir: Union[str, Path]) -> Path:
    """
    Unified Control Plane: Ensures path is safe and within the project jail.
    Prevents path traversal attacks.
    """
    root = Path(root_dir).resolve()
    raw_target = Path(file_path)
    # Resolve relative paths against the declared jail root, not the process cwd.
    target = (root / raw_target).resolve() if not raw_target.is_absolute() else raw_target.resolve()

    try:
        target.relative_to(root)
    except ValueError as exc:
        raise SecurityError(f"Access denied: Path '{file_path}' escapes project root '{root_dir}'.") from exc

    return target

def sanitize_command(cmd: List[str]):
    """
    Unified Control Plane: Validates shell commands against an allowlist.
    """
    if not cmd:
        return
    
    allowed = get_allowed_commands()
    base_cmd = os.path.basename(cmd[0])
    is_python = base_cmd.startswith("python")
    
    # We allow python, python3, python3.x etc.
    if is_python:
        # Check if it's literally "python" or "python3" or "python3.X"
        if not re.match(r"^python(\d+(\.\d+)?)?$", base_cmd):
            # Not a standard python name, check if it's in the allowed set.
            if base_cmd not in allowed:
                raise SecurityError(f"Access denied: Command '{base_cmd}' is not in the allowlist.")
    elif base_cmd not in allowed:
        raise SecurityError(f"Access denied: Command '{base_cmd}' is not in the allowlist.")
    
    # Flag dangerous flags (shallow check)
    dangerous_args = ["--eval", "-e", "--exec", "-c"]
    if is_python:
        for arg in cmd[1:]:
            if arg in dangerous_args:
                # We allow -m for pytest/compile/unittest
                if "-m" in cmd: continue
                raise SecurityError(f"Access denied: Dangerous argument '{arg}' in command.")

