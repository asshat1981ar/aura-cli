"""Error resolution engine — cache lookup, known-fix registry, AI provider fallback."""

from __future__ import annotations

import re
import time
from typing import Any, Optional

from aura.error_resolution.cache import FourLayerCache
from aura.error_resolution.safety import SafetyChecker
from aura.error_resolution.types import ResolutionConfidence, ResolutionResult


# ---------------------------------------------------------------------------
# Built-in known fixes (no AI required)
# ---------------------------------------------------------------------------

_KNOWN_FIXES: dict[str, tuple[str, str]] = {
    "ModuleNotFoundError": (
        "The required Python package is not installed.",
        "pip install {module}",
    ),
    "FileNotFoundError": (
        "The specified file or directory does not exist.",
        "Check the path and ensure the file exists before running the command.",
    ),
    "PermissionError": (
        "Insufficient permissions to access the resource.",
        "Check file permissions with 'ls -la' and adjust ownership if needed.",
    ),
    "ConnectionRefusedError": (
        "The target service is not running or is unreachable.",
        "Verify the service is started and the port/host are correct.",
    ),
    "TimeoutError": (
        "The operation timed out waiting for a response.",
        "Retry the operation, or increase the timeout limit.",
    ),
}


def _extract_module_name(error: Exception) -> str:
    """Pull the missing module name out of a ModuleNotFoundError message."""
    match = re.search(r"No module named '([^']+)'", str(error))
    return match.group(1) if match else "the_package"


def _parse_provider_response(raw: str) -> tuple[str, str, ResolutionConfidence]:
    """Parse EXPLANATION / FIX / CONFIDENCE from a provider response string."""
    explanation = ""
    fix = ""
    confidence = ResolutionConfidence.LOW

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("EXPLANATION:"):
            explanation = line[len("EXPLANATION:"):].strip()
        elif line.startswith("FIX:"):
            fix = line[len("FIX:"):].strip()
        elif line.startswith("CONFIDENCE:"):
            level = line[len("CONFIDENCE:"):].strip().lower()
            confidence = ResolutionConfidence(level) if level in ("high", "medium", "low") else ResolutionConfidence.LOW

    return explanation, fix, confidence


# ---------------------------------------------------------------------------
# Provider registry stub
# ---------------------------------------------------------------------------

class _ProviderRegistry:
    """Minimal provider registry; real AI providers are injected at runtime."""

    def get_primary(self):  # type: ignore[return]
        return None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ErrorResolutionEngine:
    """Resolve errors via cache → known-fix → AI provider pipeline."""

    def __init__(self, cache: Optional[FourLayerCache] = None) -> None:
        self.cache = cache or FourLayerCache()
        self.safety = SafetyChecker()
        self.providers = _ProviderRegistry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def resolve(
        self,
        error: Exception,
        context: Optional[dict] = None,
        auto_apply: bool = False,
    ) -> ResolutionResult:
        """Resolve *error* and return a ResolutionResult."""
        start = time.monotonic()
        cache_key = self.cache.make_key(error, context)

        # 1. Cache hit
        cached = self.cache.get(cache_key)
        if cached is not None:
            return ResolutionResult(
                original_error=str(error),
                explanation=cached.explanation,
                suggested_fix=cached.suggested_fix,
                confidence=cached.confidence,
                auto_applied=False,
                cache_hit=True,
                provider="cache",
                execution_time_ms=int((time.monotonic() - start) * 1000),
            )

        # 2. Known-fix lookup
        known = self._lookup_known_fix(error)
        if known is not None:
            explanation, fix = known
            result = ResolutionResult(
                original_error=str(error),
                explanation=explanation,
                suggested_fix=fix,
                confidence=ResolutionConfidence.HIGH,
                auto_applied=auto_apply and self.safety.is_safe_to_apply(fix),
                cache_hit=False,
                provider="known_fix",
                execution_time_ms=int((time.monotonic() - start) * 1000),
            )
            self.cache.set(cache_key, result)
            return result

        # 3. AI provider
        provider = self.providers.get_primary()
        if provider is not None:
            try:
                raw = await provider.suggest_fix(error, context)
                explanation, fix, confidence = _parse_provider_response(raw)
                should_apply = (
                    auto_apply
                    and confidence == ResolutionConfidence.HIGH
                    and self.safety.is_safe_to_apply(fix)
                )
                result = ResolutionResult(
                    original_error=str(error),
                    explanation=explanation,
                    suggested_fix=fix,
                    confidence=confidence,
                    auto_applied=should_apply,
                    cache_hit=False,
                    provider=provider.name,
                    execution_time_ms=int((time.monotonic() - start) * 1000),
                )
                self.cache.set(cache_key, result)
                return result
            except Exception:
                return ResolutionResult(
                    original_error=str(error),
                    explanation="Could not resolve the error — the AI provider failed.",
                    suggested_fix="",
                    confidence=ResolutionConfidence.LOW,
                    auto_applied=False,
                    cache_hit=False,
                    provider="error",
                    execution_time_ms=int((time.monotonic() - start) * 1000),
                )

        # 4. Fallback
        return ResolutionResult(
            original_error=str(error),
            explanation="Could not resolve the error — no provider available.",
            suggested_fix="",
            confidence=ResolutionConfidence.LOW,
            auto_applied=False,
            cache_hit=False,
            provider="none",
            execution_time_ms=int((time.monotonic() - start) * 1000),
        )

    def clear_cache(self) -> None:
        self.cache.clear()

    def get_stats(self) -> dict[str, Any]:
        return {
            "l1_cache_size": len(self.cache.l1_memory),
            "providers_available": self.providers.get_primary() is not None,
            "known_fixes_count": len(_KNOWN_FIXES),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _lookup_known_fix(self, error: Exception) -> Optional[tuple[str, str]]:
        error_type = type(error).__name__
        if error_type not in _KNOWN_FIXES:
            return None
        explanation_tmpl, fix_tmpl = _KNOWN_FIXES[error_type]
        if error_type == "ModuleNotFoundError":
            module = _extract_module_name(error)
            fix_tmpl = fix_tmpl.format(module=module)
        return explanation_tmpl, fix_tmpl
