import os
import re
from pathlib import Path
from typing import Any, List, Union
from core.exceptions import AuraError
from core.config_manager import config

class SecurityError(AuraError):
    """Raised when a security boundary is violated."""
    pass

# Hardcoded safe commands for AURA
BASE_ALLOWED_COMMANDS = {
    "python", "python3", "pytest", "git", "ls", "cat", "rm", "mkdir", "mv", "grep", "sed", "npx"
}

def get_allowed_commands() -> set:
    """Returns the effective set of allowed commands from config + base."""
    extra = config.get("security", {}).get("allowed_commands", [])
    return BASE_ALLOWED_COMMANDS.union(set(extra))

# Regex for masking secrets (best-effort)
SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{32,}", re.IGNORECASE), # Generic OpenAI/OpenRouter style
    re.compile(r"Bearer\s+[a-zA-Z0-9._-]+", re.IGNORECASE),
    re.compile(r"api[-_]?key", re.IGNORECASE) # Catch key names in dicts
]

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
    if base_cmd not in allowed:
        raise SecurityError(f"Access denied: Command '{base_cmd}' is not in the allowlist.")
    
    # Flag dangerous flags (shallow check)
    dangerous_args = ["--eval", "-e", "--exec", "-c"]
    if base_cmd == "python" or base_cmd == "python3":
        for arg in cmd[1:]:
            if arg in dangerous_args:
                # We allow -m for pytest/compile
                if "-m" in cmd: continue
                raise SecurityError(f"Access denied: Dangerous argument '{arg}' in command.")

def mask_secrets(data: Any) -> Any:
    """
    Unified Control Plane: Recursively redacts sensitive info from data.
    """
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            if any(p.search(k) for p in SECRET_PATTERNS if "api" in p.pattern or "key" in p.pattern):
                new_dict[k] = "[REDACTED]"
            else:
                new_dict[k] = mask_secrets(v)
        return new_dict
    elif isinstance(data, list):
        return [mask_secrets(i) for i in data]
    elif isinstance(data, str):
        masked = data
        for p in SECRET_PATTERNS:
            # Mask the actual secret value if found in string
            if "sk-" in p.pattern or "Bearer" in p.pattern:
                masked = p.sub("[REDACTED]", masked)
        return masked
    return data
