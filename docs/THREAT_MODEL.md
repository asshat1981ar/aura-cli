# AURA CLI Threat Model

**Version:** 1.0.0
**Date:** 2026-04-10
**Owner:** AURA Security Team
**Last Reviewed:** 2026-04-10

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        AURA CLI System                          │
│                                                                 │
│  ┌──────────┐   JWT    ┌──────────────┐   REST   ┌──────────┐  │
│  │  User /  │─────────▶│  FastAPI     │──────────▶│ Agents   │  │
│  │  CI Bot  │◀─────────│  API Server  │◀──────────│ (Planner │  │
│  └──────────┘  Bearer  │  :8001       │           │  Coder   │  │
│                        └──────┬───────┘           │  Critic) │  │
│                               │                   └────┬─────┘  │
│                        ┌──────▼───────┐                │        │
│                        │  Rate-limit  │          ┌─────▼──────┐ │
│                        │  Middleware  │          │  Sandbox   │ │
│                        │ (TokenBucket)│          │  Agent     │ │
│                        └──────┬───────┘          │ (subprocess│ │
│                               │                  │  + tempdir)│ │
│                        ┌──────▼───────┐          └────────────┘ │
│                        │  Auth DB     │                         │
│                        │  SQLite      │  ┌─────────────────┐   │
│                        │  XDG path    │  │  Memory Stack   │   │
│                        │  0600 perms  │  │  L0 Redis (opt) │   │
│                        └──────────────┘  │  L1 SQLite cache│   │
│                                          │  L2 Brain graph │   │
│                                          │  L3 JSONL log   │   │
│                                          │  Vec embeddings │   │
│                                          └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
         │                                          │
    ┌────▼─────┐                            ┌──────▼──────┐
    │  LLM API │                            │  Project FS │
    │ (OpenAI/ │                            │  (host disk)│
    │  OpenRtr)│                            └─────────────┘
    └──────────┘
```

---

## Assets Under Protection

| Asset | Classification | Location | Impact if Compromised |
|-------|---------------|----------|-----------------------|
| JWT signing key (`SECRET_KEY`) | Secret | `AURA_SECRET_KEY` env | Full auth bypass |
| API tokens / credentials | Secret | `.env`, env vars | Unauthorized API access |
| Auth database | Sensitive | `~/.local/share/aura/auth.db` | Account takeover |
| LLM API keys | Secret | `OPENAI_API_KEY` / `OPENROUTER_API_KEY` | Financial loss, quota abuse |
| Project source code | Confidential | Host filesystem | IP theft, supply-chain attack |
| Memory / decision logs | Internal | `memory/` directory | Goal/context leakage |
| Agent execution output | Internal | In-memory + brain DB | Information disclosure |
| Redis cache | Internal | `REDIS_URL` host | Cache poisoning |

---

## Threat Actors

| Actor | Motivation | Capability | Likelihood |
|-------|-----------|-----------|-----------|
| External attacker (unauthenticated) | Financial gain, disruption | Low–Medium | Medium |
| Malicious LLM output | Code injection, data exfil | Medium | Medium |
| Compromised dependency | Supply-chain attack | High | Low |
| Insider / rogue contributor | Data theft, sabotage | High | Low |
| Automated scanner / bot | Credential stuffing, DoS | Low | High |

---

## Attack Vectors & Mitigations

### 1. JWT Token Theft / Forgery

**Threat:** Attacker steals a bearer token or attempts to forge one by guessing
the signing key or exploiting weak algorithm selection.

**Controls implemented:**
- Algorithm restricted to `HS256` via `_ALLOWED_ALGORITHMS = {"HS256"}` allowlist
  in `core/auth.py`; any token presenting a different algorithm is rejected.
- 256-bit (32-byte) signing key enforced; weak keys raise `AuthError` at startup.
- JTI (JWT ID) claim issued per-token; revoked JTIs persisted in auth DB so logout
  is immediate even before expiry.
- Short expiry (15 min access / 7 day refresh) limits the stolen-token window.

**Residual risk:** HMAC secret exposure via env-var leakage. Rotate `AURA_SECRET_KEY`
immediately on suspected compromise.

---

### 2. Sandbox Escape via Raw Sockets

**Threat:** LLM-generated code bypasses the HTTP proxy env-var network block by
opening a raw `socket()` directly, reaching external hosts.

**Controls implemented:**
- Proxy env vars (`HTTP_PROXY`, `HTTPS_PROXY` → `127.0.0.1:1`) block high-level
  HTTP libraries.
- `DeprecationWarning` emitted when raw socket usage is detected in submitted code.
- Sandbox violation patterns (`_VIOLATION_PATTERNS`) log and flag socket calls.

**Residual risk / recommendation:** Proxy env vars do not block raw sockets or
non-HTTP protocols. For production workloads handling untrusted code, deploy the
sandbox inside a kernel-level network namespace (`unshare --net`) or a container
with `--network=none`. See "Known Limitations" below.

---

### 3. Proxy Bypass (Network Namespace Limitation)

**Threat:** Sandbox relies on proxy environment variables for network isolation.
Code that does not use the `requests`/`urllib` stack (e.g., raw sockets, `ctypes`
calling `connect()`) can bypass the proxy.

**Controls implemented:**
- Defense-in-depth: `PYTHONPATH=""`, `PYTHONNOUSERSITE=1` reduce available attack
  surface.
- Violation-detection regex patterns log suspicious imports (`socket`, `ctypes`).

**Known limitation (accepted):** This is a documented, accepted risk. The sandbox
is designed as a *pre-apply smoke test* for LLM-generated code, not a full security
boundary for arbitrary untrusted code execution. See "Known Limitations" section.

**Recommendation:** Use `unshare --net` or Docker `--network=none` when running the
sandbox in environments that execute genuinely untrusted third-party code.

---

### 4. Command Injection

**Threat:** LLM-generated shell commands contain payloads (e.g., `; rm -rf /`,
backtick expansion) that execute arbitrary OS commands.

**Controls implemented:**
- `core/sanitizer.sanitize_command()` applies an allowlist of safe command patterns
  before any shell execution; non-matching commands raise `SecurityError`.
- Denylist includes destructive patterns: `rm -rf /`, `mkfs.*`, `reboot`,
  `dd if=`, `:(){ :|:& };:` (fork bomb), etc.
- `SecurityError` is caught by the sandbox agent, logged, and returned as a failed
  `SandboxResult` — execution never reaches the shell.

**Residual risk:** Allowlist completeness. Review and extend `sanitize_command()`
when new command categories are added to agent workflows.

---

### 5. Auth Database Compromise

**Threat:** Attacker gains read/write access to the SQLite auth database to extract
password hashes, inject admin accounts, or revoke legitimate tokens.

**Controls implemented:**
- Database stored under the XDG data directory (`~/.local/share/aura/auth.db` by
  default), outside the project repository.
- File permissions set to `0600` (owner read/write only) at creation time.
- Passwords hashed with `bcrypt` via `passlib`; raw passwords never persisted.
- `AURA_AUTH_DB_PATH` env var allows relocation to encrypted volumes in production.

**Recommendation:** Include auth DB in regular encrypted backups. For multi-user
deployments, store on an encrypted filesystem or use a dedicated secrets manager.

---

### 6. Algorithm Confusion Attacks

**Threat:** Attacker crafts a JWT with `"alg": "none"` or switches to RS256 with a
forged public key to bypass HMAC signature verification.

**Controls implemented:**
- `_ALLOWED_ALGORITHMS = {"HS256"}` in `core/auth.py` is passed directly to the
  `jose.jwt.decode()` call; any other algorithm value raises `JWTError`.
- `algorithm="none"` guard: the `jose` library rejects `none` when an algorithms
  allowlist is provided.
- No asymmetric key material is present in the deployment, eliminating RS256/ES256
  confusion vectors.

---

### 7. Denial of Service

**Threat:** Attacker floods the API with requests to exhaust server threads, memory,
or downstream LLM API quota.

**Controls implemented:**
- Token-bucket rate limiter (`aura_cli/middleware/rate_limit.py`) enforces per-path,
  per-user limits:

  | Endpoint | Limit | Window |
  |----------|-------|--------|
  | `/api/v1/auth/login` | 5 req | 60 s |
  | `/api/v1/auth/refresh` | 20 req | 60 s |
  | `/execute`, `/run` | 10 req | 60 s |
  | `/webhook/goal` | 30 req | 60 s |
  | (default) | 100 req | 60 s |

- Burst allowance (`burst = requests // 5`) absorbs short spikes without false
  positives.
- Sandbox resource limits: `RLIMIT_CPU` 30 s, `RLIMIT_AS` 512 MiB per execution.
- Concurrent sandbox loop limit prevents runaway agent self-correction cycles.
- Stale bucket cleanup every 5 minutes prevents memory growth from idle clients.

**Residual risk:** Rate limiter is in-process (`InMemoryRateLimiter`); state is lost
on restart and not shared across multiple API server replicas. For multi-replica
deployments, migrate to a Redis-backed limiter.

---

## Security Controls Summary

| Control | Implementation | Location | Status |
|---------|---------------|----------|--------|
| JWT signing (HS256 only) | `_ALLOWED_ALGORITHMS` allowlist | `core/auth.py` | ✅ Active |
| JTI revocation | SQLite revoked-JTI store | `core/auth.py` | ✅ Active |
| bcrypt password hashing | `passlib.CryptContext` | `core/auth.py` | ✅ Active |
| Token-bucket rate limiting | `InMemoryRateLimiter` | `aura_cli/middleware/rate_limit.py` | ✅ Active |
| Sandbox process isolation | `subprocess` + `tempfile.TemporaryDirectory` | `agents/sandbox.py` | ✅ Active |
| Sandbox network blocking | Proxy env vars → `127.0.0.1:1` | `agents/sandbox.py` | ✅ Active |
| Sandbox resource limits | `RLIMIT_CPU` / `RLIMIT_AS` | `agents/sandbox.py` | ✅ Active |
| Command sanitizer allowlist | `sanitize_command()` + `SecurityError` | `core/sanitizer.py` | ✅ Active |
| Auth DB file permissions | `0600` + XDG path | `core/auth.py` | ✅ Active |
| Algorithm confusion guard | `algorithms=["HS256"]` in decode | `core/auth.py` | ✅ Active |
| Secret scanning (CI) | `scripts/scan_secrets.py` | CI pipeline | ✅ Active |
| Sandbox violation logging | `aura_sandbox_violations_total` (Prometheus) | `agents/sandbox.py` | ✅ Active |

---

## Known Limitations & Accepted Risks

| ID | Limitation | Severity | Acceptance Rationale |
|----|-----------|----------|---------------------|
| L-01 | Proxy-based network blocking does not prevent raw socket connections from sandboxed code | Medium | Sandbox is a pre-apply smoke test, not a full isolation boundary; kernel namespaces recommended for untrusted code |
| L-02 | Rate limiter state is in-process; not shared across API replicas | Medium | Acceptable for single-node deployments; Redis-backed limiter planned for v1.1 multi-replica support |
| L-03 | Sandbox `open()` wrapper is Python-level only; C extensions can bypass it | Medium | Defense-in-depth measure; primary boundary is temp-directory isolation |
| L-04 | `resource` limits (`RLIMIT_*`) silently skipped on Windows | Low | Production target is Linux; documented in deployment guide |
| L-05 | Auth DB backup is a manual operational procedure | Low | Documented in production guide; automated backup tooling tracked in backlog |
| L-06 | JTI revocation list grows unbounded until manual pruning | Low | TTL-based pruning scheduled for v1.1; current volume is low |
