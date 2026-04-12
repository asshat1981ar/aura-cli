# ADR-005: Rate Limiting — Token Bucket Algorithm

**Date:** 2026-04-09
**Status:** Accepted
**Deciders:** AURA Core Team
**Tags:** security, performance, api, middleware

---

## Context

The AURA API server (`aura_cli/server.py`) is exposed over HTTP and handles
authentication, agent execution, and webhook ingestion. Without rate limiting:

- Brute-force attacks against `/api/v1/auth/login` are trivially possible.
- A single misbehaving client can monopolise sandbox execution slots and exhaust
  LLM API quota.
- Webhook floods can saturate the goal queue and cause uncontrolled memory growth.

We needed a rate-limiting strategy that:

1. Protects sensitive endpoints (login, execute) more aggressively than general ones.
2. Allows short bursts of legitimate traffic without false-positive 429s.
3. Is simple to reason about, test, and operate.
4. Can be implemented without adding infrastructure dependencies in v1.0.

### Algorithms Considered

| Algorithm | Pros | Cons |
|-----------|------|------|
| **Fixed window counter** | Simple; O(1) per request | Allows 2× burst at window boundary |
| **Sliding window log** | Precise; no boundary burst | O(n) memory per client; complex |
| **Sliding window counter** | Good precision; O(1) | Approximate at window boundaries |
| **Token bucket** ✅ | Smooth bursts; O(1); intuitive | Requires float arithmetic |
| **Leaky bucket** | Strict output rate | No burst tolerance; unfriendly UX |

### Infrastructure Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **In-process (asyncio)** ✅ | Zero dependencies; simple deploy | State lost on restart; not shared across replicas |
| **Redis-backed** | Shared across replicas; persistent | Adds Redis hard dependency; operational complexity |
| **nginx / API gateway** | Infrastructure-level; language-agnostic | Requires separate deployment; out of scope for v1.0 |

---

## Decision

Implement rate limiting as a **FastAPI middleware** using the **token bucket
algorithm** with an **in-process async store** (`InMemoryRateLimiter`).

### Token Bucket Design

```
  ┌─────────────────────────────────────────────────────┐
  │  TokenBucket per (endpoint, user_id | IP)           │
  │                                                     │
  │  capacity  = limit.requests + limit.burst           │
  │  tokens    = capacity  (on bucket creation)         │
  │  refill    = limit.requests / limit.window_seconds  │
  │              (tokens/second, continuous)            │
  │                                                     │
  │  On each request:                                   │
  │    elapsed   = now - last_refill                    │
  │    tokens   += elapsed * refill_rate                │
  │    tokens    = min(tokens, capacity)                │
  │    if tokens >= 1.0:                                │
  │        tokens -= 1.0; ALLOW                         │
  │    else:                                            │
  │        retry_after = (1.0 - tokens) / refill_rate   │
  │        DENY (HTTP 429)                              │
  └─────────────────────────────────────────────────────┘
```

### Per-Endpoint Limits

| Endpoint | `requests` | `window_s` | `burst` |
|----------|-----------|-----------|--------|
| `/api/v1/auth/login` | 5 | 60 | 1 |
| `/api/v1/auth/refresh` | 20 | 60 | 4 |
| `/execute` | 10 | 60 | 2 |
| `/run` | 10 | 60 | 2 |
| `/webhook/goal` | 30 | 60 | 6 |
| `(default)` | 100 | 60 | 20 |

`burst = max(1, requests // 5)` when not specified explicitly.

### Key Identity

Rate limit keys are scoped to `{endpoint_path}:{user_id}` for authenticated
requests, and `{endpoint_path}:{client_ip}` (with `X-Forwarded-For` support) for
unauthenticated requests. Health-check paths (`/health`, `/ready`, `/api/health`)
are exempt.

### Response Headers

Every response carries:
- `X-RateLimit-Limit`: configured request limit
- `X-RateLimit-Window`: window in seconds
- `Retry-After` (on 429 only): seconds until the bucket refills enough to allow 1 request

### Bucket Lifecycle

Inactive buckets (last activity > 1 hour) are pruned during a cleanup pass that
runs at most once every 5 minutes to prevent unbounded memory growth from idle
or one-time clients.

---

## Consequences

### Positive

- Zero new infrastructure dependencies; the limiter runs inside the existing
  asyncio event loop with a single `asyncio.Lock` for thread safety.
- Token bucket provides a smooth, user-friendly burst tolerance — legitimate
  clients making a short flurry of requests are not immediately 429'd.
- Endpoint-specific limits allow aggressive protection of the login endpoint
  without penalising normal API usage.
- `Retry-After` header enables well-behaved clients to back off automatically.
- All limit configuration is in a single dict (`ENDPOINT_LIMITS` in
  `aura_cli/middleware/rate_limit.py`), making tuning straightforward.

### Negative / Risks

- **State lost on restart**: buckets reset when the process restarts; an attacker
  can bypass limits by triggering a restart. Acceptable for v1.0 single-node
  deployments.
- **Not shared across replicas**: in a multi-replica deployment each instance
  maintains its own buckets, allowing up to N× the intended limit where N is the
  replica count. Tracked as limitation L-02 in the threat model.
- **In-memory only**: bucket state is not observable (no admin endpoint to inspect
  current fill levels). Planned for v1.1 observability sprint.

### Follow-on Actions

- [ ] **v1.1**: Migrate to a Redis-backed distributed token bucket when multi-replica
  support is required (see ADR-002 for Redis optionality contract).
- [ ] **v1.1**: Expose a read-only `/admin/rate-limits` endpoint for operational
  visibility.
- [ ] **v1.1**: Add Prometheus gauge `aura_rate_limit_tokens_remaining` per bucket.
- [ ] Review and tighten `/execute` and `/run` limits once production traffic
  patterns are observed.
