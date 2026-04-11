"""Safety classifier for error-resolution suggested fixes."""

from __future__ import annotations

import re


# Commands that are safe to auto-apply without human review.
_SAFE_PATTERNS: list[str] = [
    r"^git\s+add\b",
    r"^git\s+commit\b",
    r"^git\s+status\b",
    r"^git\s+diff\b",
    r"^git\s+log\b",
    r"^git\s+fetch\b",
    r"^git\s+pull\b",
    r"^git\s+checkout\b(?!.*--force)",
    r"^pip\s+install\b",
    r"^pip3\s+install\b",
    r"^python\s+-m\s+pip\s+install\b",
    r"^mkdir\b",
    r"^touch\b",
    r"^pytest\b",
    r"^python\s+-m\s+pytest\b",
    r"^docker\s+build\b",
    r"^npm\s+install\b",
    r"^npm\s+ci\b",
    r"^echo\b",
    r"^cat\b",
    r"^ls\b",
    r"^pwd\b",
    r"^which\b",
]

# Commands that must never be auto-applied.
_DANGEROUS_PATTERNS: list[str] = [
    r"\brm\s+.*-[^\s]*r[^\s]*f\b",      # rm -rf / rm -fr
    r"\brm\s+.*-[^\s]*f[^\s]*r\b",
    r"^sudo\b",
    r"\bsudo\s",
    r"^dd\b",
    r"\bdd\s+if=",
    r"\bmkfs\b",
    r"\bformat\s",
    r"DROP\s+TABLE",
    r"DROP\s+DATABASE",
    r"TRUNCATE\s+TABLE",
    r"\|\s*(ba)?sh\b",                   # curl | sh / wget | bash
    r"\|\s*bash\b",
    r">\s*/dev/",
    r":\(\)\{.*\}",                      # fork bomb pattern
]

# Commands that need human review but aren't outright dangerous.
_SENSITIVE_PATTERNS: list[str] = [
    r"^rm\b",
    r"\bgit\s+push\s+.*(-f|--force)\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bgit\s+clean\b",
    r"\bdocker\s+system\s+prune\b",
    r"\bdocker\s+rm\b",
    r"\bkubectl\s+delete\b",
    r"\bkubectl\s+drain\b",
    r"^\s*kill\b",
    r"^\s*pkill\b",
    r"\bchmod\s+777\b",
]


class SafetyChecker:
    """Classify shell commands by safety level for auto-apply decisions."""

    def get_safety_level(self, command: str) -> str:
        """Return 'safe', 'sensitive', or 'dangerous'."""
        if not command or not command.strip():
            return "sensitive"

        cmd = command.strip()

        for pattern in _DANGEROUS_PATTERNS:
            if re.search(pattern, cmd, re.IGNORECASE):
                return "dangerous"

        for pattern in _SAFE_PATTERNS:
            if re.search(pattern, cmd, re.IGNORECASE):
                return "safe"

        for pattern in _SENSITIVE_PATTERNS:
            if re.search(pattern, cmd, re.IGNORECASE):
                return "sensitive"

        return "sensitive"

    def is_safe_to_apply(self, command: str) -> bool:
        """Return True only for commands classified as 'safe'."""
        return self.get_safety_level(command) == "safe"

    def explain_safety(self, command: str) -> str:
        """Return a human-readable explanation of the safety classification."""
        level = self.get_safety_level(command)
        if level == "safe":
            return f"'{command}' is safe to auto-apply — it is a well-known, low-risk operation."
        if level == "dangerous":
            return (
                f"'{command}' is dangerous — it is destructive or irreversible and must never "
                "be auto-applied without explicit human confirmation."
            )
        return (
            f"'{command}' requires human review before applying — it may have side effects "
            "or be difficult to reverse."
        )
