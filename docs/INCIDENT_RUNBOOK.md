# AURA Incident Response Runbook

> **Audience:** On-call engineers  
> **Updated:** Sprint 8  
> **Related alerts:** `infra/prometheus/alerts.yml`  
> **Dashboard:** Grafana → *AURA Overview*

---

## Table of Contents

1. [Severity Levels](#severity-levels)
2. [On-Call Escalation Path](#on-call-escalation-path)
3. [Alert Runbooks](#alert-runbooks)
   - [AuraAPIDown](#aura-api-down)
   - [AuraHighErrorRate](#aura-high-error-rate)
   - [AuraSlowResponse](#aura-slow-response)
   - [AuraGoalQueueBacklog](#aura-goal-queue-backlog)
   - [AuraMemoryHigh](#aura-memory-high)
4. [Post-Incident Review Checklist](#post-incident-review-checklist)

---

## Severity Levels

| Level | Name     | Definition | Response SLA | Example |
|-------|----------|------------|-------------|---------|
| **P0** | Critical | Complete service outage; all users impacted; data loss risk | Immediate (< 5 min) | AURA API totally unreachable |
| **P1** | Major    | Core functionality broken for a significant subset of users | < 15 min | Error rate >20 %, p95 latency > 10 s |
| **P2** | Moderate | Degraded performance or partial feature loss; workaround exists | < 1 hour | p95 latency 2–10 s; queue backlog building |
| **P3** | Minor    | Non-blocking issue; user-visible but low blast radius | < 4 hours | Memory usage creeping up; dashboard gap |

---

## On-Call Escalation Path

```
Alert fires
    │
    ▼
Primary on-call (15 min SLA)
    │  no ack / unable to resolve
    ▼
Secondary on-call / team lead (30 min SLA)
    │  still unresolved or P0/P1
    ▼
Engineering manager + CTO (P0 only, immediate)
```

**Communication channels:**
- P0/P1 → `#incidents` Slack channel (auto-created by PagerDuty)  
- P2/P3 → `#platform-alerts` Slack channel  
- Status page: update within 10 min of P0/P1 declaration

---

## Alert Runbooks

### AuraAPIDown {#aura-api-down}

**Severity:** P0  
**Alert expression:** `up{job=~"aura.*"} == 0` for 1 m

#### Symptoms
- Prometheus fires `AuraAPIDown`
- `/health` endpoint returns non-200 or connection refused
- Web UI shows error / blank page
- All API calls fail with 5xx or timeout

#### Investigation Steps

1. **Confirm the container is running:**
   ```bash
   docker ps --filter name=aura-api
   docker logs aura-api --tail=100
   ```

2. **Check health endpoint directly:**
   ```bash
   curl -v http://localhost:8001/health
   ```

3. **Inspect resource limits (OOM kill?):**
   ```bash
   docker inspect aura-api | grep -A5 OOMKilled
   dmesg | grep -i "oom\|killed" | tail -20
   ```

4. **Check disk space (SQLite can fail if full):**
   ```bash
   df -h
   docker system df
   ```

5. **Review startup errors:**
   ```bash
   docker logs aura-api 2>&1 | grep -i "error\|exception\|traceback"
   ```

#### Remediation

| Root Cause | Fix |
|------------|-----|
| Container crashed (OOM) | `docker restart aura-api`; then increase memory limit in `docker-compose.prod.yml` |
| Config / env missing | Verify `.env.n8n` is present; re-deploy with `docker compose up -d api` |
| Disk full | Clear old logs: `docker system prune -f`; expand volume |
| Port conflict | `lsof -i :8001`; kill conflicting process |
| Code bug after deploy | Roll back: `docker compose -f docker-compose.prod.yml up -d --image ghcr.io/asshat1981ar/aura-cli:<prev-version>` |

---

### AuraHighErrorRate {#aura-high-error-rate}

**Severity:** P1 (>20 %) / P2 (5–20 %)  
**Alert expression:** `rate(aura_pipeline_runs_total{status="failed"}[5m]) / rate(aura_pipeline_runs_total[5m]) > 0.05` for 5 m

#### Symptoms
- High proportion of pipeline run failures in Grafana *Error Rate* panel
- Clients receiving 5xx responses or error JSON bodies

#### Investigation Steps

1. **Identify failing goals:**
   ```bash
   curl -s http://localhost:8001/api/v1/goals/queue | python3 -m json.tool | grep -i status
   ```

2. **Check structured logs for error events:**
   ```bash
   docker logs aura-api 2>&1 | python3 -c "
   import sys, json
   for line in sys.stdin:
       try:
           r = json.loads(line)
           if r.get('level') in ('ERROR', 'WARN'):
               print(json.dumps(r, indent=2))
       except: pass
   " | tail -60
   ```

3. **Inspect specific run errors in Prometheus:**
   ```promql
   topk(5, rate(aura_pipeline_runs_total{status="failed"}[5m]))
   ```

4. **Check external dependencies (LLM provider, MCP servers):**
   ```bash
   curl -v https://api.anthropic.com/v1/messages -H "x-api-key: $ANTHROPIC_API_KEY" --max-time 5
   ```

#### Remediation

| Root Cause | Fix |
|------------|-----|
| LLM API quota exceeded | Check provider dashboard; switch model adapter or increase quota |
| Broken tool/plugin | Check `aura_sandbox_violations_total` metric; disable offending tool |
| Bad deployment | Roll back to previous image tag |
| Transient upstream failure | Monitor; alert will self-resolve when upstream recovers |

---

### AuraSlowResponse {#aura-slow-response}

**Severity:** P2  
**Alert expression:** `histogram_quantile(0.95, sum by (le, instance)(rate(http_request_duration_seconds_bucket[5m]))) > 2` for 5 m

#### Symptoms
- Grafana *p95 Latency* panel shows >2 s
- Users reporting slow tool execution in web UI

#### Investigation Steps

1. **Identify slow endpoints:**
   ```promql
   histogram_quantile(0.95,
     sum by (le, endpoint) (
       rate(http_request_duration_seconds_bucket[5m])
     )
   )
   ```

2. **Check for blocking I/O or CPU saturation:**
   ```bash
   docker stats aura-api --no-stream
   top -b -n1 -p $(docker inspect -f '{{.State.Pid}}' aura-api)
   ```

3. **Check DB response times (SQLite lock contention):**
   ```bash
   docker exec aura-api python3 -c "
   import time, sqlite3
   s = time.time()
   sqlite3.connect('/data/aura_auth.db').execute('SELECT 1')
   print(f'SQLite round-trip: {(time.time()-s)*1000:.1f} ms')
   "
   ```

4. **Review active pipeline runs:**
   ```bash
   curl -s http://localhost:8001/api/v1/status | python3 -m json.tool
   ```

#### Remediation

| Root Cause | Fix |
|------------|-----|
| LLM call latency | Enable streaming; add response timeout `AURA_RUN_TOOL_TIMEOUT_S` |
| SQLite lock contention | Switch to WAL mode; reduce concurrent pipelines |
| CPU throttle (Docker limit) | Increase `cpus` in `docker-compose.prod.yml` |
| Slow MCP server dependency | Timeout failing MCP calls; investigate target MCP server |

---

### AuraGoalQueueBacklog {#aura-goal-queue-backlog}

**Severity:** P2  
**Alert expression:** `aura_goal_queue_depth > 50` for 10 m

#### Symptoms
- Goal queue depth gauge rising in Grafana
- Webhook `/api/v1/goal` submissions succeed but are never processed
- `active_pipeline_runs` is 0 or stuck

#### Investigation Steps

1. **Check queue status via API:**
   ```bash
   curl -s -H "Authorization: Bearer $AURA_API_KEY" \
     http://localhost:8001/api/v1/goals/queue | python3 -m json.tool
   ```

2. **Check if orchestrator loop is running:**
   ```bash
   docker logs aura-api 2>&1 | grep -i "orchestrator\|loop\|goal" | tail -30
   ```

3. **Check for stuck running items (never completing):**
   ```bash
   curl -s -H "Authorization: Bearer $AURA_API_KEY" \
     http://localhost:8001/api/v1/goals/queue | python3 -c "
   import sys, json, time
   q = json.load(sys.stdin)
   now = time.time()
   for g in q.get('goals', []):
       if g['status'] == 'running':
           age = now - g.get('started_at', now)
           print(f\"{g['goal_id']}: running {age:.0f}s\")
   "
   ```

#### Remediation

| Root Cause | Fix |
|------------|-----|
| Orchestrator stalled | `docker restart aura-api` (queue is in-memory; goals will be lost — check if persistence is needed) |
| Single goal blocking queue | Use API to cancel stalled goal; implement per-goal timeout |
| Webhook flood / abuse | Inspect caller; apply rate limiting at nginx layer |
| Queue not draining (resource limit) | Scale out horizontally or increase `AURA_RUN_TOOL_TIMEOUT_S` |

> ⚠️ **Note:** `_webhook_goal_queue` is currently in-memory. A restart will clear all pending goals. Ensure webhook callers implement retry logic.

---

### AuraMemoryHigh {#aura-memory-high}

**Severity:** P2 (>512 MB) / P1 (>1.5 GB — approaching container 2 GB limit)  
**Alert expression:** `process_resident_memory_bytes{job=~"aura.*"} > 536870912` for 5 m

#### Symptoms
- RSS rising over time without returning to baseline (memory leak)
- Container may be OOM-killed if left unchecked

#### Investigation Steps

1. **Confirm current RSS:**
   ```bash
   docker stats aura-api --no-stream --format "table {{.MemUsage}}"
   ```

2. **Profile memory in running container:**
   ```bash
   docker exec aura-api python3 -c "
   import tracemalloc, json
   tracemalloc.start()
   # ... attach to live objects if debugpy is available
   snap = tracemalloc.take_snapshot()
   top = snap.statistics('lineno')
   for s in top[:10]:
       print(s)
   "
   ```

3. **Check `memory/brain_v2.db` size:**
   ```bash
   docker exec aura-api du -sh /data/
   ```

4. **Check large in-memory structures (goal archive, memory store):**
   ```bash
   docker logs aura-api 2>&1 | grep -i "memory\|cache\|loaded" | tail -20
   ```

#### Remediation

| Root Cause | Fix |
|------------|-----|
| Memory leak in LLM response cache | Restart API; open bug; reduce cache TTL |
| Growing goal archive in RAM | Implement pagination / pruning in `GoalQueue` |
| Large memory store loaded on startup | Lazy-load or paginate `memory_store.read_log()` |
| Legitimate load growth | Increase container memory limit in `docker-compose.prod.yml` |

---

## Post-Incident Review Checklist

Complete within **48 hours** of P0/P1 resolution; 1 week for P2.

### 5-Why Template

| # | Why? | Root Cause Layer |
|---|------|-----------------|
| 1 | Why did the incident occur? | Proximate cause |
| 2 | Why was the proximate cause not prevented? | Missing safeguard |
| 3 | Why was the safeguard missing? | Process gap |
| 4 | Why did the process gap exist? | Systemic issue |
| 5 | Why does the systemic issue persist? | Cultural / org |

### Review Checklist

- [ ] **Timeline** documented (UTC): alert fired → detected → acknowledged → resolved
- [ ] **Impact** quantified: # users affected, duration, data loss (if any)
- [ ] **Root cause** identified and agreed upon by team
- [ ] **5-Why analysis** completed (link to doc)
- [ ] **Immediate fix** merged and deployed
- [ ] **Follow-up tickets** created in backlog for systemic fixes
  - [ ] Alert threshold tuned (if too noisy / too late)
  - [ ] Runbook updated with any new investigation steps found
  - [ ] Monitoring gap identified? New metric/alert added
- [ ] **Notification sent** to affected stakeholders (if P0/P1)
- [ ] **Status page** updated to resolved
- [ ] **Meeting scheduled** for team blameless post-mortem (P0/P1 only)
- [ ] **Action item owners** assigned with due dates
