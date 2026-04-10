"""Parser for AI provider responses."""

import re
from typing import Tuple

from .types import ResolutionConfidence, ResolutionResult


class ResponseParser:
    """Parse AI provider responses into structured ResolutionResult."""
    
    # Common patterns for parsing structured responses
    EXPLANATION_PATTERNS = [
        r"EXPLANATION:\s*(.+?)(?=\n\s*(?:FIX|SUGGESTION|CONFIDENCE):|$)",
        r"Explanation:\s*(.+?)(?=\n\s*(?:Fix|Suggestion|Confidence):|$)",
        r"## Explanation\s*\n(.+?)(?=\n##|\Z)",
    ]
    
    FIX_PATTERNS = [
        r"FIX:\s*(.+?)(?=\n\s*(?:CONFIDENCE|EXPLANATION):|$)",
        r"Suggestion:\s*(.+?)(?=\n\s*(?:Confidence|Explanation):|$)",
        r"## Fix\s*\n(.+?)(?=\n##|\Z)",
        r"## Solution\s*\n(.+?)(?=\n##|\Z)",
        r"```(?:bash|sh|shell)?\s*\n(.+?)```",
    ]
    
    CONFIDENCE_PATTERNS = [
        r"CONFIDENCE:\s*(high|medium|low)",
        r"Confidence:\s*(high|medium|low)",
    ]
    
    def parse(
        self,
        response: str,
        original_error: Exception,
        provider: str,
        execution_time_ms: int,
    ) -> ResolutionResult:
        """Parse AI response into ResolutionResult.
        
        Args:
            response: Raw AI response text
            original_error: The original exception
            provider: Provider name ("openai", "ollama", etc.)
            execution_time_ms: Time taken to get response
            
        Returns:
            Structured ResolutionResult
        """
        explanation = self._extract_explanation(response)
        fix = self._extract_fix(response)
        confidence = self._extract_confidence(response)
        
        # Fallbacks if parsing fails
        if not explanation:
            explanation = f"Error: {str(original_error)}"
        
        if not fix:
            # Use entire response as fix if no specific format
            fix = response.strip()
        
        return ResolutionResult(
            original_error=str(original_error),
            explanation=explanation.strip(),
            suggested_fix=fix.strip(),
            confidence=confidence,
            auto_applied=False,
            cache_hit=False,
            provider=provider,
            execution_time_ms=execution_time_ms,
        )
    
    def _extract_explanation(self, response: str) -> str:
        """Extract explanation from response."""
        for pattern in self.EXPLANATION_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return ""
    
    def _extract_fix(self, response: str) -> str:
        """Extract fix/suggestion from response."""
        for pattern in self.FIX_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return ""
    
    def _extract_confidence(self, response: str) -> ResolutionConfidence:
        """Extract confidence level from response."""
        for pattern in self.CONFIDENCE_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                level = match.group(1).lower()
                try:
                    return ResolutionConfidence(level)
                except ValueError:
                    pass
        
        # Default to low if not specified or unrecognized
        return ResolutionConfidence.LOW


class KnownFixesRegistry:
    """Registry of known fixes for common errors.
    
    Acts as L3 cache - curated solutions for frequently occurring errors.
    """
    
    def __init__(self):
        self._fixes: dict[str, Tuple[str, str, ResolutionConfidence]] = {}
        self._load_builtin_fixes()
    
    def _load_builtin_fixes(self):
        """Load built-in known fixes."""
        # ImportError: No module named 'xyz'
        self._fixes["ModuleNotFoundError"] = (
            "The required Python module is not installed.",
            "pip install <module_name>",
            ResolutionConfidence.HIGH,
        )
        
        # Git: not a git repository
        self._fixes["git not a git repository"] = (
            "Not inside a git repository. Initialize one or navigate to a repo.",
            "git init",
            ResolutionConfidence.HIGH,
        )
        
        # Port already in use
        self._fixes["address already in use"] = (
            "Another process is using the required port.",
            "lsof -i :<port> && kill -9 <pid>",
            ResolutionConfidence.MEDIUM,
        )
        
        # Permission denied
        self._fixes["permission denied"] = (
            "Insufficient permissions to access the file or directory.",
            "chmod +x <file>  # or use sudo for system files",
            ResolutionConfidence.MEDIUM,
        )
        
        # Connection refused
        self._fixes["connection refused"] = (
            "Could not connect to the service. It may not be running.",
            "# Start the service first, e.g.,\nservice <name> start",
            ResolutionConfidence.MEDIUM,
        )
        
        # Docker daemon not running
        self._fixes["cannot connect to the docker daemon"] = (
            "Docker daemon is not running.",
            "sudo systemctl start docker  # Linux\n# or open Docker Desktop  # Mac/Windows",
            ResolutionConfidence.HIGH,
        )
        
        # Python syntax error
        self._fixes["SyntaxError"] = (
            "Python syntax error in the code.",
            "# Check for:\n# - Missing colons (:)\n# - Mismatched brackets/parentheses\n# - Incorrect indentation",
            ResolutionConfidence.MEDIUM,
        )
        
        # Merge conflict
        self._fixes["merge conflict"] = (
            "Git merge conflicts need to be resolved.",
            "# 1. Edit files to resolve conflicts\n# 2. git add <files>\n# 3. git commit",
            ResolutionConfidence.HIGH,
        )
    
    def lookup(self, error: Exception) -> ResolutionResult | None:
        """Look up a known fix for an error.
        
        Args:
            error: The exception to look up
            
        Returns:
            ResolutionResult if found, None otherwise
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Try exact error type match
        if error_type in self._fixes:
            return self._make_result(error, error_type)
        
        # Try substring match
        for key in self._fixes:
            if key.lower() in error_str:
                return self._make_result(error, key)
        
        return None
    
    def _make_result(self, error: Exception, key: str) -> ResolutionResult:
        """Create ResolutionResult from registry entry."""
        explanation, fix, confidence = self._fixes[key]
        return ResolutionResult(
            original_error=str(error),
            explanation=explanation,
            suggested_fix=fix,
            confidence=confidence,
            auto_applied=False,
            cache_hit=True,
            provider="known_fix",
            execution_time_ms=0,
        )
    
    def add_fix(
        self,
        error_pattern: str,
        explanation: str,
        fix: str,
        confidence: ResolutionConfidence = ResolutionConfidence.MEDIUM,
    ):
        """Add a new known fix to the registry.
        
        Args:
            error_pattern: Pattern to match against error messages
            explanation: Human-readable explanation
            fix: Suggested fix
            confidence: Confidence level
        """
        self._fixes[error_pattern] = (explanation, fix, confidence)
