# ADR-002: Redis Optionality Contract

**Date:** 2025-01-01  
**Status:** Accepted  
**Deciders:** AURA Core Team  
**Tags:** infrastructure, storage, concurrency, redis, sqlite

---

## Context

### Current State: SQLite-only

AURA's persistent state currently lives in three SQLite databases:

| Database | File | Used by |
|---|---|---|
| Brain / memory store | `memory/brain_v2.db` | `memory/brain.py` — `Brain` class (WAL mode, threading.Lock) |
| Goal queue | `memory/goal_queue.json` | `core/goal_queue.py` — `GoalQueue` class (JSON file, deque) |
| Context graph | `memory/context_graph.db` | `memory/context/` — semantic memory |

`Brain` already uses SQLite WAL mode and a `threading.Lock` (`_db_lock` context manager, `memory/brain.py` lines 35–38, 48–50) to serialise writes within a single process. This works correctly for single-process execution.

### The Problem: Multi-agent Write Contention

When multiple agent subprocesses run concurrently—as happens during swarm runs (`AURA_ENABLE_SWARM=1`), parallel plan decomposition, or when the API server and a background CLI process are both active—SQLite hits its single-writer limit:

- `SQLITE_BUSY` / "database is locked" errors under concurrent writes.
- `GoalQueue` uses a plain JSON file; two processes writing simultaneously will corrupt it.
- The in-memory `_webhook_goal_queue` dict in `aura_cli/server.py` (line 601) is **not durable**: a server restart silently drops all queued goals.
- Pub/sub (notifying one agent when another completes a goal) is not possible with SQLite.

### Why Not Just Require Redis?

AURA runs on Android/Termux, low-power development laptops, and CI containers where Redis is not available. Mandating Redis would break single-developer usage and increase on-boarding friction substantially. The single-process SQLite path must remain fully functional and be the **default**.

---

## Decision

Redis is **optional**. Its presence is controlled by two environment variables:

```bash
REDIS_URL=redis://localhost:6379/0   # connection string
REDIS_ENABLED=false                  # explicit opt-in gate (default: false)
```

When `REDIS_ENABLED=false` (or `REDIS_URL` is unset), AURA behaves exactly as today — single-writer SQLite mode with no external dependency.

When `REDIS_ENABLED=true` and `REDIS_URL` is set and reachable, Redis serves as:
1. **Write serialiser** for `GoalQueue` — replace the JSON file with a Redis List (`LPUSH` / `RPOPLPUSH` / `LREM`).
2. **Pub/sub bus** — goal lifecycle events (`goal.queued`, `goal.started`, `goal.completed`, `goal.failed`) published to a Redis channel so any subscriber (agent, WebSocket client, n8n) can react.
3. **Durable webhook goal queue** — replace the in-memory `_webhook_goal_queue` dict in `aura_cli/api/routers/runs.py` with a Redis Hash keyed by `goal_id`.

SQLite (`brain_v2.db`) is **not replaced by Redis**. It remains the source of truth for memory, embeddings, and the knowledge graph. Redis is purely an inter-process coordination layer.

### Design Principle: Transparent Degradation

Every Redis call site must be wrapped in a `try/except` that falls back to SQLite mode:

```python
# Pseudocode pattern — required at every Redis call site
try:
    if redis_client and redis_client.ping():
        return redis_client.rpoplpush(...)
except (redis.ConnectionError, redis.TimeoutError):
    _log_redis_degradation()
    # fall through to SQLite path

# SQLite fallback (always present)
return goal_queue_sqlite.next()
```

A `REDIS_DEGRADATION` log event must be emitted exactly once per reconnect cycle (not once per call) to avoid log spam.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | _(unset)_ | Redis connection URL. Supports `redis://`, `rediss://` (TLS), `redis+unix://` (socket). |
| `REDIS_ENABLED` | `false` | Explicit opt-in. Redis is not used even if `REDIS_URL` is set, unless this is `true`. |

Both must be set to enable Redis:
```bash
REDIS_URL=redis://localhost:6379/0
REDIS_ENABLED=true
```

### `aura.config.json` equivalent

The same settings can be expressed in `aura.config.json`:
```json
{
  "redis_url": "redis://localhost:6379/0",
  "redis_enabled": false
}
```

Environment variables take precedence over the config file (standard AURA override chain).

---

## Consequences

### Positive
- Single-developer usage on Termux/Android is unaffected — no Redis required.
- Multi-agent concurrency scales safely when Redis is available.
- `GoalQueue` becomes crash-safe across processes (Redis List survives process death; WAL already handles this for Brain).
- WebSocket clients and n8n workflows can subscribe to goal lifecycle events without polling.
- The fallback path exercises the same code paths as the Redis path in tests (just with a different backend), preventing silent divergence.

### Negative
- Two code paths to maintain for every persistence operation.
- Redis connection management (pool, health checks, reconnect) adds ~150 lines of new infrastructure code.
- Developers testing the Redis path locally must run `redis-server` (or `docker run redis`).

### Neutral
- `brain_v2.db` is not changed. Read-heavy semantic search remains on SQLite, which is appropriate.

---

## Rollback Plan

If Redis becomes unavailable during operation (network partition, OOM kill, planned maintenance):

1. **Automatic:** All Redis call sites catch `redis.ConnectionError` / `redis.TimeoutError` and fall back to the SQLite/file path immediately. No human intervention required for continued operation.
2. **In-flight goals:** Goals already popped from the Redis List but not yet acknowledged (`RPOPLPUSH` source → processing list) are recovered on the next startup via `GoalQueue.recover()` — the existing in-flight recovery mechanism (see `core/goal_queue.py` lines 81–99).
3. **Pub/sub subscribers** simply stop receiving events; they do not crash. The polling fallback (`GET /webhook/status/{goal_id}`) remains functional.
4. **Logging:** A single `REDIS_UNAVAILABLE` structured log event is emitted with the connection error and the fallback path taken.
5. **Recovery:** When Redis becomes reachable again, the connection pool re-establishes automatically on the next operation. No restart required.

There is no data loss: SQLite WAL is the authoritative store.

---

## Code Locations That Need Updating

The following files must be modified when Redis support is implemented:

### New files to create

| File | Purpose |
|---|---|
| `core/redis_client.py` | Connection pool factory, health check, `get_redis()` helper |
| `core/goal_queue_redis.py` | Redis-backed `GoalQueue` implementation (same interface as `GoalQueue`) |

### Existing files to modify

| File | Change |
|---|---|
| `core/goal_queue.py` | Wrap `_save_queue` / `next` / `complete` / `fail` with Redis adapter calls when enabled |
| `core/config_manager.py` | Add `redis_url` and `redis_enabled` to `DEFAULT_CONFIG` and `_KEY_VALIDATORS` |
| `aura_cli/api/routers/runs.py` | Replace `_webhook_goal_queue: Dict` (from `server.py` line 601) with Redis Hash adapter |
| `aura_cli/entrypoint.py` | Call `redis_client.ping()` on startup; log warning if REDIS_ENABLED=true but unreachable |
| `.env.example` | Document `REDIS_URL` and `REDIS_ENABLED` (see ADR task s0-env-audit) |

### pyproject.toml

Add `redis` as an optional dependency:
```toml
[project.optional-dependencies]
redis = ["redis>=5.0"]
```

Install with: `pip install -e ".[redis]"`

---

## Testing Strategy

| Test type | Description |
|---|---|
| Unit (no Redis) | `REDIS_ENABLED=false` — all existing tests pass unchanged |
| Unit (mock Redis) | Use `fakeredis` library to test Redis code paths without a live server |
| Integration | Docker Compose fixture starts `redis:7-alpine`; `REDIS_ENABLED=true` is set for the test session |
| Chaos | Kill Redis mid-test; assert fallback to SQLite path without exception propagation |

---

## Alternatives Considered

| Option | Rejected reason |
|---|---|
| **Always require Redis** | Breaks single-developer Termux/Android use case; increases CI complexity |
| **PostgreSQL SKIP LOCKED** | Much heavier than Redis for a queue; no pub/sub |
| **SQLite + WAL + retry loop** | Does not solve multi-process write contention, only reduces collision window |
| **NATS/RabbitMQ** | Overkill; Redis List + pub/sub covers all current needs |

---

## Related

- ADR-001: api\_server.py Decomposition (replaces in-memory `_webhook_goal_queue`)
- `core/goal_queue.py` — current file-based implementation
- `memory/brain.py` — SQLite Brain (not replaced, complementary)
- `REDIS_URL`, `REDIS_ENABLED` env vars — documented in `.env.example`
