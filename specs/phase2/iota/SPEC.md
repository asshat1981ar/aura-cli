# IOTA Specification: AI-Powered Error Resolution

> **Agent**: IOTA  
> **Feature**: AI Error Resolution  
> **Priority**: P1 (Critical Path)  
> **Duration**: 3 days (Week 1: Mon-Wed)  
> **Depends On**: BETA (error_presenter from Phase 1)

---

## Overview

IOTA provides AI-powered error resolution for AURA CLI commands. When a command fails, IOTA analyzes the error, queries AI providers (OpenAI, Ollama) for solutions, caches results for performance, and can optionally auto-apply safe fixes.

## Goals

1. Capture errors from CLI commands
2. Query AI providers for resolution suggestions
3. Cache resolutions in 4-layer cache (memory, disk, known fixes, AI)
4. Present solutions to users via error_presenter
5. Auto-apply safe fixes (optional, user-configurable)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Error Resolution Flow                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│   Command Error                                              │
│       │                                                      │
│       ▼                                                      │
│   ┌──────────────┐                                          │
│   │ ErrorCapture │  Capture error context                   │
│   └──────┬───────┘                                          │
│          │                                                   │
│          ▼                                                   │
│   ┌─────────────────────────────────────┐                   │
│   │        4-Layer Cache Check          │                   │
│   │  1. L1: In-memory cache             │                   │
│   │  2. L2: Disk cache (SQLite)         │                   │
│   │  3. L3: Known fixes registry        │                   │
│   │  4. L4: AI provider query           │                   │
│   └──────────────┬──────────────────────┘                   │
│                  │                                           │
│       Cache Hit ◄┴─► Cache Miss                              │
│          │              │                                    │
│          ▼              ▼                                    │
│   ┌──────────┐   ┌──────────────┐                          │
│   │ Return   │   │ Query AI     │                          │
│   │ Cached   │   │ Provider     │                          │
│   │ Solution │   └──────┬───────┘                          │
│   └────┬─────┘          │                                   │
│        │                ▼                                   │
│        │         ┌──────────────┐                          │
│        │         │ Parse &      │                          │
│        │         │ Validate     │                          │
│        │         └──────┬───────┘                          │
│        │                │                                   │
│        └────────────────┘                                   │
│                         │                                   │
│                         ▼                                   │
│   ┌─────────────────────────────────────┐                   │
│   │     Present to User (Rich UI)       │                   │
│   │  - Error explanation                │
│   │  - Suggested fix                    │
│   │  - Auto-apply option (if safe)      │                   │
│   └─────────────────────────────────────┘                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

```
aura/error_resolution/
├── __init__.py              # Public API exports
├── engine.py                # Core resolution engine
├── cache.py                 # 4-layer cache implementation
├── providers.py             # AI provider abstractions
├── known_fixes.py           # Registry of known solutions
├── parser.py                # AI response parsing
├── safety.py                # Safety checks for auto-apply
└── types.py                 # Type definitions
```

## Interface Design

### Public API

```python
# aura/error_resolution/__init__.py

from .engine import ErrorResolutionEngine
from .types import ResolutionResult, ResolutionConfidence

__all__ = [
    "ErrorResolutionEngine",
    "resolve_error",           # Convenience function
    "ResolutionResult",
    "ResolutionConfidence",
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
        context: Additional context (command, cwd, env)
        auto_apply: Whether to auto-apply safe fixes
        
    Returns:
        ResolutionResult with solution and metadata
    """
    engine = ErrorResolutionEngine()
    return await engine.resolve(error, context, auto_apply)
```

### Core Types

```python
# aura/error_resolution/types.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional

class ResolutionConfidence(Enum):
    HIGH = "high"       # Verified fix, safe to auto-apply
    MEDIUM = "medium"   # Likely correct, user confirmation recommended
    LOW = "low"         # Suggestion, manual review required

@dataclass
class ResolutionResult:
    original_error: str
    explanation: str
    suggested_fix: str
    confidence: ResolutionConfidence
    auto_applied: bool
    cache_hit: bool
    provider: str       # "openai", "ollama", "cache", "known_fix"
    execution_time_ms: int
```

### Engine

```python
# aura/error_resolution/engine.py

class ErrorResolutionEngine:
    def __init__(self):
        self.cache = FourLayerCache()
        self.providers = ProviderRegistry()
        self.known_fixes = KnownFixesRegistry()
        self.safety = SafetyChecker()
    
    async def resolve(
        self,
        error: Exception,
        context: dict | None = None,
        auto_apply: bool = False,
    ) -> ResolutionResult:
        # 1. Check cache layers
        cached = self.cache.get(error, context)
        if cached:
            return self._apply_if_safe(cached, auto_apply)
        
        # 2. Check known fixes
        known = self.known_fixes.lookup(error)
        if known:
            self.cache.set(error, known)
            return self._apply_if_safe(known, auto_apply)
        
        # 3. Query AI provider
        provider = self.providers.get_primary()
        suggestion = await provider.suggest_fix(error, context)
        
        # 4. Parse and validate
        result = self._parse_suggestion(suggestion)
        
        # 5. Cache result
        self.cache.set(error, result)
        
        # 6. Apply if safe and requested
        return self._apply_if_safe(result, auto_apply)
```

## 4-Layer Cache

```python
# aura/error_resolution/cache.py

class FourLayerCache:
    """
    L1: In-memory LRU cache (fastest, ephemeral)
    L2: Disk cache via SQLite (persistent, local)
    L3: Known fixes registry (curated solutions)
    L4: AI provider (slowest, most capable)
    """
    
    def __init__(self):
        self.l1_memory = LRUCache(maxsize=100)
        self.l2_disk = SQLiteCache(path="~/.aura/error_cache.db")
        # L3 and L4 are separate components
    
    def get(self, error: Exception, context: dict) -> Optional[ResolutionResult]:
        key = self._make_key(error, context)
        
        # Try L1
        if key in self.l1_memory:
            return self.l1_memory[key]
        
        # Try L2
        result = self.l2_disk.get(key)
        if result:
            self.l1_memory[key] = result  # Promote to L1
            return result
        
        return None
    
    def set(self, error: Exception, result: ResolutionResult):
        key = self._make_key(error, {})
        self.l1_memory[key] = result
        self.l2_disk.set(key, result, ttl=86400 * 7)  # 7 days
```

## AI Providers

```python
# aura/error_resolution/providers.py

from abc import ABC, abstractmethod

class AIProvider(ABC):
    @abstractmethod
    async def suggest_fix(
        self,
        error: Exception,
        context: dict | None,
    ) -> str:
        """Return raw AI suggestion."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

class OpenAIProvider(AIProvider):
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
    
    async def suggest_fix(self, error: Exception, context: dict | None) -> str:
        prompt = self._build_prompt(error, context)
        # OpenAI API call with retry logic from Phase 1
        return await self._call_api(prompt)

class OllamaProvider(AIProvider):
    def __init__(self, host: str = "http://localhost:11434", model: str = "codellama"):
        self.host = host
        self.model = model
    
    async def suggest_fix(self, error: Exception, context: dict | None) -> str:
        prompt = self._build_prompt(error, context)
        # Ollama local API call
        return await self._call_ollama(prompt)

class ProviderRegistry:
    def __init__(self):
        self._providers: dict[str, AIProvider] = {}
        self._primary: str | None = None
    
    def register(self, name: str, provider: AIProvider, primary: bool = False):
        self._providers[name] = provider
        if primary:
            self._primary = name
    
    def get_primary(self) -> AIProvider:
        if self._primary:
            return self._providers[self._primary]
        # Auto-select: prefer Ollama (local), fallback to OpenAI
        if "ollama" in self._providers:
            return self._providers["ollama"]
        return self._providers.get("openai")
```

## Integration with Error Presenter

```python
# aura_cli/error_presenter.py (extension)

from aura.error_resolution import resolve_error, ResolutionConfidence

async def present_error_with_resolution(
    error: Exception,
    verbose: bool = False,
    suggest_fix: bool = True,
):
    """Present error with AI-powered resolution suggestion."""
    
    # Present original error
    console = Console(stderr=True)
    console.print(Panel(str(error), title="Error", border_style="red"))
    
    if suggest_fix:
        with console.status("[yellow]Analyzing error..."):
            result = await resolve_error(error)
        
        # Display resolution
        style = {
            ResolutionConfidence.HIGH: "green",
            ResolutionConfidence.MEDIUM: "yellow",
            ResolutionConfidence.LOW: "red",
        }[result.confidence]
        
        console.print(Panel(
            f"[bold]{result.explanation}[/bold]\n\n"
            f"Suggested fix:\n[code]{result.suggested_fix}[/code]",
            title=f"AI Suggestion ({result.confidence.value} confidence)",
            border_style=style,
        ))
        
        if result.auto_applied:
            console.print("[green]✓ Fix automatically applied[/green]")
```

## Safety for Auto-Apply

```python
# aura/error_resolution/safety.py

class SafetyChecker:
    """Determines if a fix can be safely auto-applied."""
    
    # Safe command patterns (read-only or easily reversible)
    SAFE_PATTERNS = [
        r"^git\s+add\s+",
        r"^git\s+commit\s+",
        r"^git\s+stash\s+",
        r"^pip\s+install\s+",
        r"^npm\s+install\s+",
        r"^mkdir\s+",
        r"^touch\s+",
    ]
    
    # Dangerous patterns (never auto-apply)
    DANGEROUS_PATTERNS = [
        r"rm\s+-rf",
        r"dd\s+if=",
        r">\s+/dev/",
        r"sudo\s+",
        r"DROP\s+TABLE",
        r"DELETE\s+FROM",
    ]
    
    def is_safe_to_apply(self, command: str) -> bool:
        # Check dangerous first
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return False
        
        # Check safe patterns
        for pattern in self.SAFE_PATTERNS:
            if re.match(pattern, command, re.IGNORECASE):
                return True
        
        # Default: not safe
        return False
```

## Configuration

```yaml
# aura.config.json additions
{
  "error_resolution": {
    "enabled": true,
    "auto_apply_safe": false,
    "cache_ttl_days": 7,
    "providers": {
      "primary": "ollama",
      "ollama": {
        "host": "http://localhost:11434",
        "model": "codellama"
      },
      "openai": {
        "model": "gpt-4o-mini",
        "timeout_seconds": 30
      }
    }
  }
}
```

## CLI Commands

```bash
# Enable/disable error resolution
aura config set error_resolution.enabled true

# Configure provider
aura config set error_resolution.providers.primary openai

# Test error resolution
aura doctor --test-error-resolution

# Clear error cache
aura error-resolution clear-cache
```

## Test Strategy

```python
# tests/unit/error_resolution/test_engine.py

class TestErrorResolutionEngine:
    async def test_cache_hit_returns_cached_result(self):
        """L1 cache should return result without calling provider."""
        
    async def test_known_fix_lookup(self):
        """Should check known fixes registry before AI."""
        
    async def test_openai_provider_call(self):
        """Should query OpenAI when cache miss."""
        
    async def test_ollama_provider_call(self):
        """Should query Ollama when configured."""
        
    async def test_safety_checker_blocks_dangerous(self):
        """Should never auto-apply dangerous commands."""
        
    async def test_safety_checker_allows_safe(self):
        """Should auto-apply safe commands when enabled."""
        
    async def test_result_parsing(self):
        """Should parse AI response into structured result."""
```

## Success Criteria

- [ ] Error resolution engine functional
- [ ] 4-layer cache working (L1-L4)
- [ ] OpenAI provider implemented
- [ ] Ollama provider implemented
- [ ] Safety checker prevents dangerous auto-apply
- [ ] 15+ unit tests passing
- [ ] Integration with error_presenter complete
- [ ] Documentation complete

---

*Specification created: 2026-04-10*  
*Review: Upon implementation start*
