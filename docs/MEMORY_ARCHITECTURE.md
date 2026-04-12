# AURA Memory Architecture

**Version:** 1.0.0
**Date:** 2026-04-10
**Owner:** AURA Platform Team

---

## Overview

AURA uses a multi-tier memory system designed to balance read latency, persistence
durability, and semantic retrieval quality. Each tier serves a distinct purpose and
the system degrades gracefully when optional tiers are unavailable (e.g., Redis not
configured).

The memory stack is rooted in the `memory/` package and consumed by agents at
runtime through the `Brain` facade (`memory/brain.py`).

---

## Memory Tiers

```
  ┌─────────────────────────────────────────────────────┐
  │                  Agent / API Request                │
  └──────────────────────────┬──────────────────────────┘
                             │  recall(query)
                             ▼
  ┌──────────────────────────────────────────────────────┐
  │  L0: Redis Cache  (optional — REDIS_URL env var)    │
  │  • In-memory key/value, sub-millisecond reads        │
  │  • TTL-controlled; evicted on expiry or restart      │
  │  • Adapter: memory/redis_cache_adapter.py            │
  └──────────────────────────┬───────────────────────────┘
                   miss │    │ hit → return
                        ▼
  ┌──────────────────────────────────────────────────────┐
  │  L1: Local SQLite Cache  (always present)           │
  │  • File: memory/local_cache.db                       │
  │  • Persists across restarts; TTL via updated_at col  │
  │  • Adapter: memory/local_cache_adapter.py            │
  └──────────────────────────┬───────────────────────────┘
                   miss │    │ hit → populate L0, return
                        ▼
  ┌──────────────────────────────────────────────────────┐
  │  L2: SQLite Semantic Graph  (always present)        │
  │  • File: memory/brain.db  (schema v5)                │
  │  • NetworkX graph lazily loaded for relate() calls   │
  │  • Tables: memory, weaknesses, kv_store,             │
  │            vector_store_data, innovation_sessions    │
  │  • Module: memory/brain.py (Brain class)             │
  └──────────────────────────┬───────────────────────────┘
                   miss │    │ hit → populate L1+L0, return
                        ▼
  ┌──────────────────────────────────────────────────────┐
  │  L3: JSONL Decision Log  (append-only)              │
  │  • File: memory/decision_log.jsonl (rotated @ 10 MB) │
  │  • Rotations: .jsonl.1 … .jsonl.3 kept              │
  │  • Write-only from query path; read by analytics     │
  │  • Module: memory/store.py (MemoryStore class)       │
  └──────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────┐
  │  Vector Store  (cross-cutting)                      │
  │  • Embeddings stored in brain.db:vector_store_data   │
  │  • Module: memory/vector_store_v2.py (VectorStoreV2) │
  │  • Backed by core/vector_store.py                    │
  │  • Similarity search via cosine distance on BLOBs    │
  └──────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

### Read Path

```
Agent calls brain.recall(query)
        │
        ├─▶ Check in-process _recall_cache (5 s TTL, invalidated on remember())
        │       └─ HIT: return immediately
        │
        ├─▶ L0 Redis (if REDIS_URL set)
        │       └─ HIT: populate _recall_cache, return
        │
        ├─▶ L1 local_cache.db
        │       └─ HIT: populate L0 + _recall_cache, return
        │
        ├─▶ L2 brain.db  (memory table, full-text match)
        │       └─ HIT: populate L1 + L0 + _recall_cache, return
        │
        └─▶ Vector Store (semantic similarity fallback)
                └─ Results sorted by cosine similarity, populate upper tiers
```

### Write Path

```
Agent calls brain.remember(content)
        │
        ├─▶ INSERT into brain.db:memory
        │
        ├─▶ Embed content → INSERT into brain.db:vector_store_data
        │
        ├─▶ Invalidate _recall_cache (flush dict)
        │
        ├─▶ Invalidate L1 entry (if key-addressable)
        │
        ├─▶ Invalidate / update L0 Redis (if connected)
        │
        └─▶ (Cycle completion) MemoryStore.append_log(decision_entry)
                                └─▶ Rotate if decision_log.jsonl > 10 MB
```

---

## Brain SQLite Schema

**File:** `memory/brain.db`
**Schema version:** 5 (stored in `schema_version` table)
**WAL mode** enabled; `PRAGMA synchronous=NORMAL`, `busy_timeout=5000 ms`.

```sql
-- Version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

-- Primary episodic memory
CREATE TABLE IF NOT EXISTS memory (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT
);
CREATE INDEX IF NOT EXISTS idx_memory_id ON memory (id);

-- Self-identified weaknesses from critic agent
CREATE TABLE IF NOT EXISTS weaknesses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT,
    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Embedding blobs for semantic search
CREATE TABLE IF NOT EXISTS vector_store_data (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    content   TEXT,
    embedding BLOB
);

-- Generic key-value store (config, counters, state flags)
CREATE TABLE IF NOT EXISTS kv_store (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- Innovation Catalyst session state
CREATE TABLE IF NOT EXISTS innovation_sessions (
    session_id          TEXT PRIMARY KEY,
    problem_statement   TEXT NOT NULL,
    status              TEXT DEFAULT 'active',
    current_phase       TEXT DEFAULT 'immersion',
    phases_completed    TEXT,   -- JSON array
    techniques          TEXT,   -- JSON array
    constraints         TEXT,   -- JSON object
    ideas_generated     INTEGER DEFAULT 0,
    ideas_selected      INTEGER DEFAULT 0,
    output_data         TEXT,   -- JSON-serialised InnovationOutput
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at          DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_innovation_sessions_status
    ON innovation_sessions (status);
CREATE INDEX IF NOT EXISTS idx_innovation_sessions_updated
    ON innovation_sessions (updated_at);
```

**NetworkX graph** (`Brain._graph`): lazily initialised on first `relate()` call;
nodes are memory `id` values, edges carry a `weight` float. The graph is held
in-process and rebuilt from `brain.db` after a restart.

---

## Decision Log Format

**File:** `memory/decision_log.jsonl`
**Encoding:** UTF-8, one JSON object per line.
**Rotation:** file rotated at 10 MB (`AURA_LOG_MAX_BYTES`); 3 rotations kept
(`.jsonl.1`, `.jsonl.2`, `.jsonl.3`).

Each entry is written by `MemoryStore.append_log()` at the end of an agent cycle:

| Field | Type | Description |
|-------|------|-------------|
| `cycle_id` | string | Unique identifier for the agent execution cycle (e.g. `cycle_0bd55aca9d00`) |
| `goal` | string | Full goal text passed to the agent |
| `goal_type` | string | Goal category (`feature`, `bugfix`, `refactor`, …) |
| `phase_outputs` | object | Per-phase results keyed by phase name |
| `dry_run` | boolean | Whether the cycle ran in dry-run mode |
| `beads` | array | Intermediate reasoning steps / bead chain |
| `stop_reason` | string | Why the cycle terminated (`success`, `max_retries`, `error`) |
| `started_at` | string | ISO-8601 UTC timestamp of cycle start |
| `completed_at` | string | ISO-8601 UTC timestamp of cycle end |
| `duration_s` | number | Wall-clock duration in seconds |
| `outcome` | string | Final outcome label (`passed`, `failed`, `partial`) |
| `cycle_summary` | string | Human-readable one-line summary of what was accomplished |

---

## Cache Invalidation Strategy

| Tier | Invalidation Trigger | Method |
|------|---------------------|--------|
| In-process `_recall_cache` | Any `brain.remember()` write | Full dict flush |
| L0 Redis | `remember()` write to matching key | `DEL key` via adapter |
| L1 SQLite cache | `remember()` write to matching key | `DELETE WHERE key=?` |
| L2 brain.db | Never invalidated; append-only memory rows | — |
| Vector store | Embedding model change or DB corruption | `rebuild()` on `VectorStoreV2` |

The in-process recall cache has a hard TTL of 5 seconds (`Brain._cache_ttl`),
acting as a backstop so stale reads cannot persist beyond one polling interval even
if an explicit invalidation is missed.

---

## Cold Start Behavior

When AURA starts with an empty or missing database:

1. `Brain.__init__` creates `memory/brain.db` and runs `CREATE TABLE IF NOT EXISTS`
   for all tables; schema version is set to 5.
2. L0 Redis connection is attempted; if `REDIS_URL` is absent or unreachable the
   tier is silently skipped and all reads fall through to L1.
3. L1 `local_cache.db` is created empty if absent.
4. `decision_log.jsonl` is created on first `append_log()` call.
5. The NetworkX graph (`_graph`) is `None` until the first `relate()` call; agents
   that only call `remember()` / `recall()` never pay the NetworkX import cost
   (~885 ms on first load).
6. Vector store is functional immediately; an empty embedding table yields zero
   similarity results, causing the system to fall back to keyword-based `recall()`.

---

## Performance Characteristics

| Operation | Typical Latency | Notes |
|-----------|----------------|-------|
| `recall()` — in-process cache hit | < 0.1 ms | Dict lookup, no I/O |
| `recall()` — L0 Redis hit | 0.5–2 ms | Network round-trip to localhost Redis |
| `recall()` — L1 SQLite hit | 1–5 ms | File I/O, WAL read |
| `recall()` — L2 brain.db full scan | 5–50 ms | Scales with row count; index on `id` helps |
| `recall()` — vector similarity search | 20–200 ms | Depends on embedding count and model |
| `remember()` — write + embed | 50–500 ms | Dominated by embedding model latency |
| `append_log()` — JSONL append | < 1 ms | Single file write, no lock |
| `append_log()` — with rotation | 5–20 ms | File rename + prune of old rotations |
| Brain cold start | 200–400 ms | SQLite schema creation + PRAGMA setup |
| NetworkX graph first load | ~885 ms | Lazy import; paid once per process |

**Scalability notes:**
- `brain.db:memory` scans degrade past ~30 k rows without a content-indexed FTS
  table; a full-text search index (`FTS5`) is planned for v1.1.
- Redis L0 cache should be used in production to keep API response times below
  100 ms for repeated recall queries.
- Decision log rotation keeps disk usage bounded at approximately 4 × 10 MB = 40 MB
  regardless of cycle count.
