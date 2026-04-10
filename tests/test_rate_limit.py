"""Comprehensive tests for aura_cli/middleware/rate_limit.py

Covers:
  - RateLimit dataclass fields and burst defaults
  - TokenBucket token consumption and time-based refill
  - 200 responses within limit, 429 when exceeded
  - Retry-After and X-RateLimit-* headers
  - Per-endpoint limits (login=5, execute=10)
  - IP-based vs user-based keying
  - Exempt paths (/health, /ready, /api/health)
  - Limit reset after window expires
  - Multiple users with independent buckets
  - Concurrent async requests
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from aura_cli.middleware.rate_limit import (
    ENDPOINT_LIMITS,
    InMemoryRateLimiter,
    RateLimit,
    TokenBucket,
    get_rate_limit_key,
    rate_limit_middleware,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app() -> FastAPI:
    """Minimal FastAPI app wired with rate_limit_middleware."""
    app = FastAPI()
    app.add_middleware(BaseHTTPMiddleware, dispatch=rate_limit_middleware)

    @app.get("/api/v1/auth/login")
    async def login():
        return {"ok": True}

    @app.get("/execute")
    async def execute():
        return {"ok": True}

    @app.get("/some/path")
    async def some_path():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/ready")
    async def ready():
        return {"ok": True}

    @app.get("/api/health")
    async def api_health():
        return {"ok": True}

    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_limiter(monkeypatch):
    """Replace the module-level limiter with a fresh one before each test."""
    import aura_cli.middleware.rate_limit as rl_mod
    monkeypatch.setattr(rl_mod, "_limiter", InMemoryRateLimiter())


@pytest.fixture
def app():
    return _make_app()


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=True)


# ============================================================================
# 1. RateLimit dataclass
# ============================================================================

class TestRateLimitDataclass:
    def test_fields_are_set_correctly(self):
        rl = RateLimit(requests=10, window_seconds=60)
        assert rl.requests == 10
        assert rl.window_seconds == 60

    def test_burst_defaults_to_requests_div_5(self):
        rl = RateLimit(requests=10, window_seconds=60)
        assert rl.burst == 2  # max(1, 10 // 5)

    def test_burst_minimum_is_1_for_small_limits(self):
        rl = RateLimit(requests=4, window_seconds=60)
        assert rl.burst == 1  # max(1, 4 // 5) = max(1, 0) = 1

    def test_explicit_burst_is_respected(self):
        rl = RateLimit(requests=10, window_seconds=60, burst=7)
        assert rl.burst == 7


# ============================================================================
# 2. TokenBucket unit tests
# ============================================================================

class TestTokenBucket:
    def test_consume_full_bucket_is_allowed(self):
        rl = RateLimit(requests=5, window_seconds=60)
        bucket = TokenBucket(tokens=5.0)
        allowed, retry = bucket.consume(rl)
        assert allowed is True
        assert retry == pytest.approx(0.0)

    def test_consume_empty_bucket_is_denied(self):
        rl = RateLimit(requests=5, window_seconds=60)
        bucket = TokenBucket(tokens=0.0)
        allowed, retry = bucket.consume(rl)
        assert allowed is False
        assert retry > 0.0

    def test_token_bucket_refills_over_time(self):
        """Backdating last_refill simulates time passing without real sleep."""
        rl = RateLimit(requests=60, window_seconds=60)  # 1 token/sec
        bucket = TokenBucket(tokens=0.0)

        # Empty bucket is denied
        allowed, _ = bucket.consume(rl)
        assert not allowed

        # Simulate 3 seconds elapsing
        bucket.last_refill -= 3.0
        allowed, _ = bucket.consume(rl)
        assert allowed is True

    def test_tokens_are_capped_at_requests_plus_burst(self):
        """A very long idle period should not push tokens past the max."""
        rl = RateLimit(requests=5, window_seconds=60, burst=2)
        bucket = TokenBucket(tokens=0.0)
        bucket.last_refill -= 1_000.0  # massive elapsed time

        bucket.consume(rl)  # triggers refill then consumes one
        assert bucket.tokens == pytest.approx(float(rl.requests + rl.burst) - 1.0)

    def test_limit_resets_after_window_expires(self):
        """After a full window passes, a depleted bucket should allow again."""
        rl = RateLimit(requests=5, window_seconds=60)
        bucket = TokenBucket(tokens=0.0)

        allowed, _ = bucket.consume(rl)
        assert not allowed

        bucket.last_refill -= 60.0  # full window elapsed
        allowed, _ = bucket.consume(rl)
        assert allowed is True


# ============================================================================
# 3. Middleware: requests within/beyond limit
# ============================================================================

class TestMiddlewareBasic:
    def test_requests_within_limit_return_200(self, client):
        """All requests within the limit should receive 200."""
        for _ in range(5):  # login limit = 5
            r = client.get("/api/v1/auth/login")
            assert r.status_code == 200

    def test_exceeding_limit_returns_429(self, client):
        """The first request after limit exhaustion must return 429."""
        for _ in range(5):
            client.get("/api/v1/auth/login")
        r = client.get("/api/v1/auth/login")
        assert r.status_code == 429

    def test_retry_after_header_present_on_429(self, client):
        """429 responses must include a Retry-After header with a positive value."""
        for _ in range(5):
            client.get("/api/v1/auth/login")
        r = client.get("/api/v1/auth/login")
        assert r.status_code == 429
        assert "retry-after" in r.headers
        assert int(r.headers["retry-after"]) >= 1

    def test_429_body_contains_error_and_retry_after(self, client):
        """429 JSON body must carry 'error' and 'retry_after' keys."""
        for _ in range(5):
            client.get("/api/v1/auth/login")
        r = client.get("/api/v1/auth/login")
        assert r.status_code == 429
        data = r.json()
        assert data["error"] == "rate_limit_exceeded"
        assert "retry_after" in data
        assert data["retry_after"] > 0

    def test_ok_response_includes_ratelimit_headers(self, client):
        """Successful responses should carry X-RateLimit-Limit and X-RateLimit-Window."""
        r = client.get("/api/v1/auth/login")
        assert r.status_code == 200
        assert "x-ratelimit-limit" in r.headers
        assert "x-ratelimit-window" in r.headers


# ============================================================================
# 4. Per-endpoint limits
# ============================================================================

class TestEndpointLimits:
    def test_login_limit_is_5_per_60s(self):
        rl = ENDPOINT_LIMITS["/api/v1/auth/login"]
        assert rl.requests == 5
        assert rl.window_seconds == 60

    def test_execute_limit_is_10_per_60s(self):
        rl = ENDPOINT_LIMITS["/execute"]
        assert rl.requests == 10
        assert rl.window_seconds == 60

    def test_login_has_lower_limit_than_execute(self):
        assert (
            ENDPOINT_LIMITS["/api/v1/auth/login"].requests
            < ENDPOINT_LIMITS["/execute"].requests
        )

    def test_execute_allows_10_requests_before_429(self, client):
        for _ in range(10):
            r = client.get("/execute")
            assert r.status_code == 200
        r = client.get("/execute")
        assert r.status_code == 429

    def test_login_and_execute_have_independent_limits(self, client):
        """Exhausting the login endpoint should not affect /execute."""
        for _ in range(5):
            client.get("/api/v1/auth/login")
        # login is exhausted; /execute should still be open
        r = client.get("/execute")
        assert r.status_code == 200


# ============================================================================
# 5. Rate-limit key formatting
# ============================================================================

class TestRateLimitKey:
    def test_ip_key_used_when_no_auth(self):
        req = MagicMock()
        req.state = MagicMock(spec=[])  # no user_id attribute → AttributeError → None
        req.headers = {}
        req.client = MagicMock()
        req.client.host = "1.2.3.4"
        assert get_rate_limit_key(req) == "ip:1.2.3.4"

    def test_user_key_used_when_authenticated(self):
        req = MagicMock()
        req.state.user_id = "abc-123"
        assert get_rate_limit_key(req) == "user:abc-123"

    def test_x_forwarded_for_takes_precedence_over_client_ip(self):
        req = MagicMock()
        req.state = MagicMock(spec=[])
        req.headers = {"X-Forwarded-For": "203.0.113.5, 10.0.0.1"}
        req.client = MagicMock()
        req.client.host = "127.0.0.1"
        assert get_rate_limit_key(req) == "ip:203.0.113.5"

    def test_unknown_ip_when_no_client(self):
        req = MagicMock()
        req.state = MagicMock(spec=[])
        req.headers = {}
        req.client = None
        assert get_rate_limit_key(req) == "ip:unknown"


# ============================================================================
# 6. Multiple users have independent buckets
# ============================================================================

class TestMultipleUsers:
    def test_independent_buckets_per_user(self):
        """Exhausting user-A's bucket must not affect user-B."""

        async def _set_user_id(request: Request, call_next):
            user_id = request.headers.get("X-User-Id")
            if user_id:
                request.state.user_id = user_id
            return await call_next(request)

        # _set_user_id must run before rate_limit_middleware so add it last
        # (Starlette executes middlewares in reverse add order → last added = outermost)
        app = FastAPI()
        app.add_middleware(BaseHTTPMiddleware, dispatch=rate_limit_middleware)
        app.add_middleware(BaseHTTPMiddleware, dispatch=_set_user_id)

        @app.get("/api/v1/auth/login")
        async def login():
            return {"ok": True}

        client = TestClient(app, raise_server_exceptions=True)

        # Exhaust user-A's limit
        for _ in range(5):
            client.get("/api/v1/auth/login", headers={"X-User-Id": "user-A"})
        r_a = client.get("/api/v1/auth/login", headers={"X-User-Id": "user-A"})
        assert r_a.status_code == 429

        # User-B still has a fresh bucket
        r_b = client.get("/api/v1/auth/login", headers={"X-User-Id": "user-B"})
        assert r_b.status_code == 200


# ============================================================================
# 7. Exempt paths bypass rate limiting
# ============================================================================

class TestExemptPaths:
    @pytest.mark.parametrize("path", ["/health", "/ready", "/api/health"])
    def test_exempt_path_not_rate_limited(self, client, path):
        """Health-check paths must bypass the limiter even with a depleted bucket."""
        import aura_cli.middleware.rate_limit as rl_mod

        # Pre-fill a zero-token bucket for this path to confirm it is ignored
        rl_mod._limiter._buckets[f"{path}:ip:testclient"] = TokenBucket(tokens=0.0)

        r = client.get(path)
        assert r.status_code == 200


# ============================================================================
# 8. Concurrent requests via asyncio
# ============================================================================

async def test_concurrent_requests_respect_limit():
    """Concurrent async calls to the limiter collectively honour the limit."""
    limiter = InMemoryRateLimiter()
    rl = RateLimit(requests=5, window_seconds=60)
    key = "/api/v1/auth/login:ip:1.2.3.4"

    results = await asyncio.gather(*[limiter.check(key, rl) for _ in range(15)])

    allowed_count = sum(1 for ok, _ in results if ok)
    denied_count = sum(1 for ok, _ in results if not ok)

    # Exactly 5 initial tokens — only 5 requests should succeed
    assert allowed_count == 5
    assert denied_count == 10


async def test_concurrent_requests_via_http():
    """Concurrent HTTP requests through the full middleware stack respect limits."""
    app = _make_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        responses = await asyncio.gather(
            *[c.get("/api/v1/auth/login") for _ in range(15)]
        )

    statuses = [r.status_code for r in responses]
    assert 200 in statuses
    assert 429 in statuses
    # login limit = 5; no more than 5 should succeed
    assert statuses.count(200) <= 5
