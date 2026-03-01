"""Redaction tools for AURA to mask sensitive info in logs and outputs."""
from __future__ import annotations

import re
from typing import Any

# Regex for masking secrets (best-effort)
SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{32,}", re.IGNORECASE), # Generic OpenAI/OpenRouter style
    re.compile(r"Bearer\s+[a-zA-Z0-9._-]+", re.IGNORECASE),
    re.compile(r"api[-_]?key", re.IGNORECASE) # Catch key names in dicts
]

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
