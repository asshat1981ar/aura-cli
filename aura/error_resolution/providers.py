"""AI Provider implementations for error resolution."""

import os
from abc import ABC, abstractmethod
from typing import Optional

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    aiohttp = None  # type: ignore

from core.retry import retry_or_raise, RetryPolicy


class AIProvider(ABC):
    """Abstract base class for AI providers."""
    
    @abstractmethod
    async def suggest_fix(
        self,
        error: Exception,
        context: dict | None,
    ) -> str:
        """Request fix suggestion from AI provider.
        
        Args:
            error: The exception that occurred
            context: Additional context (command, cwd, env)
            
        Returns:
            Raw AI response string
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        pass
    
    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether the provider is available (configured, reachable)."""
        pass


class OpenAIProvider(AIProvider):
    """OpenAI API provider for error resolution."""
    
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_TIMEOUT = 30
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.timeout = timeout
        self._client: Optional[aiohttp.ClientSession] = None
    
    @property
    def name(self) -> str:
        return "openai"
    
    @property
    def available(self) -> bool:
        return self.api_key is not None
    
    async def _get_client(self):
        """Get or create HTTP client."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp is required for OpenAIProvider. Install with: pip install aiohttp")
        if self._client is None or self._client.closed:
            self._client = aiohttp.ClientSession()
        return self._client
    
    @retry_or_raise(policy=RetryPolicy(max_retries=2, base_delay=1.0))
    async def suggest_fix(self, error: Exception, context: dict | None) -> str:
        """Query OpenAI for fix suggestion."""
        if not self.api_key:
            raise RuntimeError("OpenAI API key not configured")
        
        prompt = self._build_prompt(error, context)
        
        client = await self._get_client()
        async with client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert CLI assistant. Analyze errors "
                            "and suggest specific, actionable fixes. "
                            "Respond in this format:\n"
                            "EXPLANATION: <brief explanation>\n"
                            "FIX: <specific command or steps>\n"
                            "CONFIDENCE: <high|medium|low>"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 500,
            },
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return data["choices"][0]["message"]["content"]
    
    def _build_prompt(self, error: Exception, context: dict | None) -> str:
        """Build prompt for OpenAI."""
        lines = [
            f"Error Type: {type(error).__name__}",
            f"Error Message: {str(error)}",
        ]
        
        if context:
            if context.get("command"):
                lines.append(f"Command: {context['command']}")
            if context.get("working_dir"):
                lines.append(f"Working Directory: {context['working_dir']}")
            if context.get("environment"):
                env = context["environment"]
                # Only include relevant env vars
                relevant = {k: v for k, v in env.items() if k.startswith(("AURA_", "PYTHON", "PATH"))}
                if relevant:
                    lines.append(f"Environment: {relevant}")
        
        lines.append("\nProvide a fix for this error.")
        return "\n".join(lines)
    
    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.closed:
            await self._client.close()


class OllamaProvider(AIProvider):
    """Ollama local LLM provider for error resolution."""
    
    DEFAULT_MODEL = "codellama"
    DEFAULT_HOST = "http://localhost:11434"
    DEFAULT_TIMEOUT = 60  # Local models can be slower
    
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client: Optional[aiohttp.ClientSession] = None
    
    @property
    def name(self) -> str:
        return "ollama"
    
    @property
    def available(self) -> bool:
        """Check if Ollama is reachable."""
        import asyncio
        try:
            # Quick check without async
            import urllib.request
            urllib.request.urlopen(f"{self.host}/api/tags", timeout=2)
            return True
        except:
            return False
    
    async def _get_client(self):
        """Get or create HTTP client."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp is required for OllamaProvider. Install with: pip install aiohttp")
        if self._client is None or self._client.closed:
            self._client = aiohttp.ClientSession()
        return self._client
    
    @retry_or_raise(policy=RetryPolicy(max_retries=1, base_delay=1.0))
    async def suggest_fix(self, error: Exception, context: dict | None) -> str:
        """Query Ollama for fix suggestion."""
        prompt = self._build_prompt(error, context)
        
        client = await self._get_client()
        async with client.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 500,
                },
            },
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return data["response"]
    
    def _build_prompt(self, error: Exception, context: dict | None) -> str:
        """Build prompt for Ollama."""
        lines = [
            "You are an expert CLI assistant. Analyze errors and suggest specific, actionable fixes.",
            "",
            f"Error Type: {type(error).__name__}",
            f"Error Message: {str(error)}",
        ]
        
        if context:
            if context.get("command"):
                lines.append(f"Command: {context['command']}")
            if context.get("working_dir"):
                lines.append(f"Working Directory: {context['working_dir']}")
        
        lines.extend([
            "",
            "Respond in this format:",
            "EXPLANATION: <brief explanation>",
            "FIX: <specific command or steps>",
            "CONFIDENCE: <high|medium|low>",
        ])
        
        return "\n".join(lines)
    
    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.closed:
            await self._client.close()
    
    async def list_models(self) -> list[str]:
        """List available Ollama models."""
        client = await self._get_client()
        async with client.get(f"{self.host}/api/tags") as response:
            response.raise_for_status()
            data = await response.json()
            return [m["name"] for m in data.get("models", [])]


class ProviderRegistry:
    """Registry for managing multiple AI providers."""
    
    def __init__(self):
        self._providers: dict[str, AIProvider] = {}
        self._primary: Optional[str] = None
    
    def register(
        self,
        name: str,
        provider: AIProvider,
        primary: bool = False,
    ):
        """Register a provider.
        
        Args:
            name: Provider identifier
            provider: Provider instance
            primary: Whether this is the primary provider
        """
        self._providers[name] = provider
        if primary:
            self._primary = name
    
    def get(self, name: str) -> AIProvider:
        """Get provider by name."""
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not registered")
        return self._providers[name]
    
    def get_primary(self) -> AIProvider:
        """Get the primary provider.
        
        Auto-selects if no primary is set:
        1. Ollama (local, free)
        2. OpenAI (requires API key)
        3. Raises error if none available
        """
        # If primary is set, use it
        if self._primary:
            return self._providers[self._primary]
        
        # Auto-select: prefer Ollama (local), fallback to OpenAI
        if "ollama" in self._providers and self._providers["ollama"].available:
            return self._providers["ollama"]
        
        if "openai" in self._providers and self._providers["openai"].available:
            return self._providers["openai"]
        
        raise RuntimeError(
            "No AI provider available. Configure OPENAI_API_KEY or start Ollama."
        )
    
    def list_available(self) -> list[str]:
        """List all available (configured/reachable) providers."""
        return [name for name, provider in self._providers.items() if provider.available]
    
    def create_default(self) -> "ProviderRegistry":
        """Create registry with default providers."""
        # Register OpenAI if API key exists
        openai_key = os.getenv("OPENAI_API_KEY")
        self.register("openai", OpenAIProvider(api_key=openai_key))
        
        # Register Ollama (check availability at runtime)
        self.register("ollama", OllamaProvider())
        
        return self
