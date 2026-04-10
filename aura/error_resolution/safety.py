"""Safety checks for auto-applying error resolution fixes."""

import re


class SafetyChecker:
    """Determines if a fix command can be safely auto-applied.
    
    Uses allowlists and denylists to classify commands by safety level.
    """
    
    # Safe command patterns (read-only or easily reversible)
    # These can be auto-applied when confidence is high
    SAFE_PATTERNS = [
        r"^git\s+add\s+",
        r"^git\s+commit\s+",
        r"^git\s+stash\s+",
        r"^git\s+branch\s+",
        r"^git\s+checkout\s+",
        r"^git\s+pull\s+",
        r"^git\s+fetch\s+",
        r"^git\s+merge\s+--abort",
        r"^git\s+rebase\s+--abort",
        r"^pip\s+install\s+",
        r"^pip\s+uninstall\s+-y\s+",
        r"^npm\s+install\s+",
        r"^npm\s+ci\s*",
        r"^yarn\s+install\s*",
        r"^mkdir\s+-p\s+",
        r"^mkdir\s+",
        r"^touch\s+",
        r"^cp\s+",
        r"^mv\s+",
        r"^ln\s+-s\s+",
        r"^chmod\s+[0-7]+\s+",
        r"^chown\s+\w+:\w+\s+",
        r"^docker\s+build\s+",
        r"^docker\s+pull\s+",
        r"^docker-compose\s+build\s*",
        r"^pytest\s+",
        r"^python\s+-m\s+pytest\s+",
        r"^python\s+-m\s+unittest\s+",
        r"^make\s+test\s*",
        r"^make\s+build\s*",
        r"^echo\s+",
        r"^cat\s+",
        r"^ls\s+",
        r"^pwd\s*",
        r"^cd\s+",
        r"^which\s+",
        r"^export\s+\w+=",
        r"^source\s+",
        r"^\.\s+",
        r"^unset\s+",
    ]
    
    # Dangerous patterns (never auto-apply)
    # These require explicit user confirmation
    DANGEROUS_PATTERNS = [
        r"rm\s+-rf",
        r"rm\s+-r\s+/",
        r"rm\s+-f\s+/",
        r"dd\s+if=",
        r">\s+/dev/\w+",
        r":\(\)\s*\{\s*:\|:&\s*\};:",  # Fork bomb
        r"sudo\s+",
        r"su\s+-",
        r"chmod\s+-R\s+777\s+/",
        r"chown\s+-R\s+\w+:\w+\s+/",
        r"mkfs\.",
        r"fdisk\s+",
        r"parted\s+",
        r"DROP\s+TABLE",
        r"DELETE\s+FROM\s+\w+\s+WHERE",  # Without LIMIT
        r"UPDATE\s+\w+\s+SET",  # Without WHERE
        r"INSERT\s+INTO",
        r"ALTER\s+TABLE\s+\w+\s+DROP",
        r"curl\s+.*\s*\|\s*sh",
        r"curl\s+.*\s*\|\s*bash",
        r"wget\s+.*\s*\|\s*sh",
        r"wget\s+.*\s*\|\s*bash",
        r"eval\s*\$",
        r"eval\s*\`",
        r"base64\s+.*\s*\|\s*sh",
    ]
    
    # Patterns that require extra scrutiny
    # These reduce confidence level
    SENSITIVE_PATTERNS = [
        r"rm\s+",  # Without -rf but still destructive
        r"rmdir\s+",
        r"mv\s+.*\s+/dev/null",
        r"git\s+push\s+--force",
        r"git\s+push\s+-f",
        r"git\s+reset\s+--hard",
        r"git\s+clean\s+-f",
        r"git\s+rebase",
        r"git\s+cherry-pick",
        r"docker\s+rm\s+-f",
        r"docker\s+system\s+prune",
        r"docker\s+volume\s+rm",
        r"docker-compose\s+down\s+-v",
        r"kubectl\s+delete",
        r"helm\s+delete",
    ]
    
    def is_safe_to_apply(self, command: str) -> bool:
        """Check if a command is safe to auto-apply.
        
        Args:
            command: The command to check
            
        Returns:
            True if safe to auto-apply, False otherwise
        """
        # Normalize command
        command = command.strip()
        
        # Check dangerous patterns first (immediate reject)
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return False
        
        # Check safe patterns
        for pattern in self.SAFE_PATTERNS:
            if re.match(pattern, command, re.IGNORECASE):
                return True
        
        # Default: not safe (require user confirmation)
        return False
    
    def get_safety_level(self, command: str) -> str:
        """Get detailed safety level for a command.
        
        Returns:
            "safe" - Can be auto-applied
            "sensitive" - Requires extra scrutiny
            "dangerous" - Never auto-apply
        """
        command = command.strip()
        
        # Check dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return "dangerous"
        
        # Check sensitive patterns
        for pattern in self.SENSITIVE_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return "sensitive"
        
        # Check safe patterns
        for pattern in self.SAFE_PATTERNS:
            if re.match(pattern, command, re.IGNORECASE):
                return "safe"
        
        # Default: sensitive (unknown command)
        return "sensitive"
    
    def explain_safety(self, command: str) -> str:
        """Get human-readable explanation of safety assessment."""
        level = self.get_safety_level(command)
        
        explanations = {
            "safe": "This command is considered safe and can be auto-applied.",
            "sensitive": "This command may have side effects. Please review before applying.",
            "dangerous": "This command is potentially destructive and requires explicit confirmation.",
        }
        
        return explanations.get(level, "Unknown safety level.")
