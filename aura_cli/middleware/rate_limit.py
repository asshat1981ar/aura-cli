"""
Rate limiting middleware for AURA CLI API.
Implements token bucket algorithm with per-endpoint limits.
"""
from __future__ import annotations
import time
import asyncio
from dataclasses import dataclass, field
from typing import Callable, Awaitable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

@dataclass
class RateLimit:
    requests: int
    window_seconds: int
    burst: int = 0

    def __post_init__(self):
        if self.burst == 0:
            self.burst = max(1, self.requests // 5)

@dataclass
class TokenBucket:
    tokens: float
    last_refill: float = field(default_factory=time.monotonic)

    def consume(self, limit: RateLimit) -> tuple[bool, float]:
        now = time.monotonic()
        elapsed = now - self.last_refill
        refill_rate = limit.requests / limit.window_seconds
        self.tokens = min(
            float(limit.requests + limit.burst),
            self.tokens + elapsed * refill_rate,
        )
        self.last_refill = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True, 0.0
        retry_after = (1.0 - self.tokens) / refill_rate
        return False, retry_after

class InMemoryRateLimiter:
    def __init__(self):
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()
        self._last_cleanup = time.monotonic()

    async def check(self, key: str, limit: RateLimit) -> tuple[bool, float]:
        async with self._lock:
            if key not in self._buckets:
                self._buckets[key] = TokenBucket(tokens=float(limit.requests))
            allowed, retry_after = self._buckets[key].consume(limit)
            if time.monotonic() - self._last_cleanup > 300:
                await self._cleanup()
            return allowed, retry_after

    async def _cleanup(self):
        cutoff = time.monotonic() - 3600
        expired = [k for k, v in self._buckets.items() if v.last_refill < cutoff]
        for k in expired:
            del self._buckets[k]
        self._last_cleanup = time.monotonic()

ENDPOINT_LIMITS: dict[str, RateLimit] = {
    "/api/v1/auth/login": RateLimit(requests=5, window_seconds=60),
    "/api/v1/auth/refresh": RateLimit(requests=20, window_seconds=60),
    "/execute": RateLimit(requests=10, window_seconds=60),
    "/run": RateLimit(requests=10, window_seconds=60),
    "/webhook/goal": RateLimit(requests=30, window_seconds=60),
    "default": RateLimit(requests=100, window_seconds=60),
}

_limiter = InMemoryRateLimiter()

def get_rate_limit_key(request: Request) -> str:
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        return f"user:{user_id}"
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        ip = forwarded_for.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "unknown"
    return f"ip:{ip}"

async def rate_limit_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    if request.url.path in {"/health", "/ready", "/api/health"}:
        return await call_next(request)

    limit = ENDPOINT_LIMITS.get(request.url.path, ENDPOINT_LIMITS["default"])
    key = f"{request.url.path}:{get_rate_limit_key(request)}"
    allowed, retry_after = await _limiter.check(key, limit)

    if not allowed:
        logger.warning("Rate limit exceeded", extra={"path": request.url.path})
        return JSONResponse(
            status_code=429,
            content={
                "error": "rate_limit_exceeded",
                "message": f"Too many requests. Retry after {retry_after:.1f}s",
                "retry_after": round(retry_after, 1),
            },
            headers={
                "Retry-After": str(int(retry_after) + 1),
                "X-RateLimit-Limit": str(limit.requests),
                "X-RateLimit-Window": str(limit.window_seconds),
            },
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(limit.requests)
    response.headers["X-RateLimit-Window"] = str(limit.window_seconds)
    return response
