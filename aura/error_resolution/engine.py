"""Core error resolution engine."""

import time
from typing import Optional

from .cache import FourLayerCache
from .parser import KnownFixesRegistry, ResponseParser
from .providers import ProviderRegistry
from .safety import SafetyChecker
from .types import ResolutionConfidence, ResolutionResult


class ErrorResolutionEngine:
    """Main engine for AI-powered error resolution.
    
    Orchestrates the 4-layer cache, known fixes, AI providers, and safety checks
    to provide intelligent error resolution suggestions.
    
    Example:
        >>> engine = ErrorResolutionEngine()
        >>> result = await engine.resolve(ValueError("Invalid input"))
        >>> print(result.suggested_fix)
    """
    
    def __init__(
        self,
        cache: Optional[FourLayerCache] = None,
        providers: Optional[ProviderRegistry] = None,
        safety: Optional[SafetyChecker] = None,
    ):
        """Initialize the resolution engine.
        
        Args:
            cache: 4-layer cache instance (creates default if None)
            providers: Provider registry (creates default if None)
            safety: Safety checker (creates default if None)
        """
        self.cache = cache or FourLayerCache()
        self.providers = providers or ProviderRegistry().create_default()
        self.known_fixes = KnownFixesRegistry()
        self.safety = safety or SafetyChecker()
        self.parser = ResponseParser()
    
    async def resolve(
        self,
        error: Exception,
        context: Optional[dict] = None,
        auto_apply: bool = False,
    ) -> ResolutionResult:
        """
        Resolve an error using the 4-layer resolution pipeline.
        
        Resolution pipeline:
        1. L1/L2 Cache - Check if we've seen this error before
        2. L3 Known Fixes - Check curated solutions
        3. L4 AI Provider - Query AI for suggestion
        
        Args:
            error: The exception to resolve
            context: Additional context (command, cwd, env, etc.)
            auto_apply: Whether to auto-apply safe fixes
            
        Returns:
            ResolutionResult with solution and metadata
        """
        start_time = time.time()
        
        # Create cache key
        cache_key = self.cache.make_key(error, context)
        
        # Layer 1 & 2: Check cache
        cached = self.cache.get(cache_key)
        if cached is not None:
            # Update cache hit timing
            return self._apply_if_safe(
                ResolutionResult(
                    original_error=cached.original_error,
                    explanation=cached.explanation,
                    suggested_fix=cached.suggested_fix,
                    confidence=cached.confidence,
                    auto_applied=False,
                    cache_hit=True,
                    provider="cache",
                    execution_time_ms=int((time.time() - start_time) * 1000),
                ),
                auto_apply,
            )
        
        # Layer 3: Check known fixes
        known = self.known_fixes.lookup(error)
        if known is not None:
            # Store in cache for next time
            self.cache.set(cache_key, known)
            return self._apply_if_safe(known, auto_apply)
        
        # Layer 4: Query AI provider
        provider = self.providers.get_primary()
        
        try:
            response = await provider.suggest_fix(error, context)
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Parse response
            result = self.parser.parse(
                response=response,
                original_error=error,
                provider=provider.name,
                execution_time_ms=execution_time_ms,
            )
            
            # Store in cache
            self.cache.set(cache_key, result)
            
            # Apply if safe
            return self._apply_if_safe(result, auto_apply)
            
        except Exception as e:
            # AI provider failed, return error result
            execution_time_ms = int((time.time() - start_time) * 1000)
            return ResolutionResult(
                original_error=str(error),
                explanation=f"Could not resolve: {str(e)}",
                suggested_fix="Please try again or resolve manually.",
                confidence=ResolutionConfidence.LOW,
                auto_applied=False,
                cache_hit=False,
                provider="error",
                execution_time_ms=execution_time_ms,
            )
    
    def _apply_if_safe(
        self,
        result: ResolutionResult,
        auto_apply: bool,
    ) -> ResolutionResult:
        """
        Apply the fix if it's safe to do so.
        
        Args:
            result: The resolution result
            auto_apply: Whether auto-apply is enabled
            
        Returns:
            ResolutionResult with auto_applied flag updated
        """
        if not auto_apply:
            return result
        
        # Only auto-apply high confidence results
        if result.confidence != ResolutionConfidence.HIGH:
            return result
        
        # Check safety
        if not self.safety.is_safe_to_apply(result.suggested_fix):
            return result
        
        # Try to apply (this is simplified - real implementation would execute)
        # For now, just mark as would-be-applied
        return ResolutionResult(
            original_error=result.original_error,
            explanation=result.explanation,
            suggested_fix=result.suggested_fix,
            confidence=result.confidence,
            auto_applied=True,  # Marked for application
            cache_hit=result.cache_hit,
            provider=result.provider,
            execution_time_ms=result.execution_time_ms,
        )
    
    def clear_cache(self):
        """Clear all cache layers."""
        self.cache.clear()
    
    def get_stats(self) -> dict:
        """Get engine statistics.
        
        Returns:
            Dict with cache stats and provider info
        """
        return {
            "l1_cache_size": len(self.cache.l1_memory),
            "providers_available": self.providers.list_available(),
            "known_fixes_count": len(self.known_fixes._fixes),
        }
