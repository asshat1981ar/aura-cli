"""AI-Powered Error Resolution for AURA CLI.

Provides intelligent error analysis and resolution suggestions using
AI providers (OpenAI, Ollama) with multi-layer caching.

Example:
    >>> from aura.error_resolution import resolve_error
    >>> result = await resolve_error(exception)
    >>> print(result.suggested_fix)
"""

from .engine import ErrorResolutionEngine
from .types import ResolutionResult, ResolutionConfidence
from .cache import FourLayerCache
from .providers import AIProvider, OpenAIProvider, OllamaProvider, ProviderRegistry

__all__ = [
    "ErrorResolutionEngine",
    "resolve_error",
    "ResolutionResult",
    "ResolutionConfidence",
    "FourLayerCache",
    "AIProvider",
    "OpenAIProvider",
    "OllamaProvider",
    "ProviderRegistry",
]


async def resolve_error(
    error: Exception,
    context: dict | None = None,
    auto_apply: bool = False,
) -> ResolutionResult:
    """
    Resolve an error using AI-powered suggestions.
    
    Args:
        error: The exception that occurred
        context: Additional context (command, cwd, env, etc.)
        auto_apply: Whether to auto-apply safe fixes
        
    Returns:
        ResolutionResult with solution and metadata
        
    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     result = await resolve_error(e, {"command": "deploy"})
        ...     print(f"Suggestion: {result.suggested_fix}")
    """
    engine = ErrorResolutionEngine()
    return await engine.resolve(error, context, auto_apply)
