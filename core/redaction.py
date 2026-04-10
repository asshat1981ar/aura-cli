"""Redaction tools for AURA to mask sensitive info in logs and outputs."""

from __future__ import annotations

import re
from typing import Any

# Regex for masking secrets (best-effort)
SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{32,}", re.IGNORECASE),  # Generic OpenAI/OpenRouter style
    re.compile(r"Bearer\s+[a-zA-Z0-9._-]+", re.IGNORECASE),
    re.compile(r"api[-_]?key", re.IGNORECASE),  # Catch key names in dicts
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


def redact_secrets(text: str) -> str:
    """
    Redact secrets from text string.
    
    Args:
        text: Input text that may contain secrets
        
    Returns:
        Text with secrets replaced by [REDACTED]
    """
    if not isinstance(text, str):
        return text
    
    # Additional patterns for comprehensive secret detection
    # Patterns with capture groups for prefix preservation
    patterns_with_prefix = [
        (re.compile(r'(?i)(api[_-]?key\s*[=:]\s*)["\']?[\w-]+["\']?'), 1),
        (re.compile(r'(?i)(token\s*[=:]\s*)["\']?[\w-]+["\']?'), 1),
        (re.compile(r'(?i)(password\s*[=:]\s*)["\']?[^"\'\s]+["\']?'), 1),
    ]
    
    # Patterns without capture groups (full replacement)
    patterns_full = [
        re.compile(r'ghp_[\w]{36}'),
        re.compile(r'sk-[a-zA-Z0-9]{48}'),
        re.compile(r'Bearer\s+[a-zA-Z0-9._-]+'),
    ]
    
    result = text
    
    # Handle patterns with prefix preservation
    for pattern, group_num in patterns_with_prefix:
        def replace_with_prefix(m):
            prefix = m.group(group_num) if group_num <= len(m.groups()) else ""
            return f"{prefix}[REDACTED]" if prefix else "[REDACTED]"
        result = pattern.sub(replace_with_prefix, result)
    
    # Handle patterns with full replacement
    for pattern in patterns_full:
        result = pattern.sub("[REDACTED]", result)
    
    return result
