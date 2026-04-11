# AURA CLI Production Deployment Guide

**Version:** 1.0.0  
**Last Updated:** 2026-04-09

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Deployment Options](#deployment-options)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Configuration](#configuration)
6. [Security Hardening](#security-hardening)
7. [Monitoring & Observability](#monitoring--observability)
8. [Backup & Recovery](#backup--recovery)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| Memory | 4 GB RAM | 8+ GB RAM |
| Disk | 20 GB SSD | 50+ GB SSD |
| OS | Linux (Ubuntu 22.04+) | Linux (Ubuntu 24.04 LTS) |

### Software Requirements

- Docker 24.0+ (for containerized deployment)
- Docker Compose 2.20+ (for Compose deployment)
- kubectl 1.28+ (for Kubernetes deployment)
- Python 3.11+ (for bare-metal deployment)

### Network Requirements

- Outbound HTTPS (443) for LLM APIs
- Internal port 8001 for AURA API
- Internal port 6379 for Redis (if used)
- Optional: port 9090 for Prometheus metrics

---

## Deployment Options

### 1. Docker Compose (Recommended for Single Node)

Best for: Small teams, development, single-server deployments

### 2. Kubernetes (Recommended for Production)

Best for: High availability, auto-scaling, multi-region deployments

### 3. Bare Metal / VM

Best for: Air-gapped environments, custom infrastructure

---

## Docker Deployment

### Quick Start

```bash
# Clone the repository
git clone https://github.com/asshat1981ar/aura-cli.git
cd aura-cli

# Copy environment file
cp .env.example .env

# Edit .env with your settings
# Required: AGENT_API_TOKEN, OPENAI_API_KEY or OPENROUTER_API_KEY

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Verify deployment
curl http://localhost:8001/health
```

### Production Docker Compose

```yaml
# docker-compose.prod.yml (excerpt)
version: '3.8'

services:
  api:
    image: ghcr.io/asshat1981ar/aura-cli:1.0.0
    restart: unless-stopped
    user: "1000:1000"
    security_opt:
      - no-new-privileges:true
    tmpfs:
      - /tmp
      - /run
    mem_limit: 512m
    cpus: '1.0'
    environment:
      - AURA_ENV=production
      - AURA_LOG_LEVEL=info
      - AGENT_API_TOKEN=${AGENT_API_TOKEN}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
    volumes:
      - aura-memory:/app/memory
      - aura-logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7.2-alpine
    restart: unless-stopped
    security_opt:
      - no-new-privileges:true
    user: "999:999"
    mem_limit: 256m
    cpus: '0.5'
    volumes:
      - redis-data:/data

volumes:
  aura-memory:
  aura-logs:
  redis-data:
```

### Security Hardening (Docker)

The production compose file includes:

- `no-new-privileges:true` — Prevents privilege escalation
- `user:` — Runs as non-root user
- `tmpfs` — Ephemeral /tmp and /run
- `mem_limit` / `cpus` — Resource constraints
- `read_only` — Immutable root filesystem (where applicable)
- Logging limits — Prevents disk exhaustion

---

## Kubernetes Deployment

### Namespace Setup

```bash
kubectl create namespace aura
kubectl config set-context --current --namespace=aura
```

### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: aura-config
data:
  AURA_ENV: "production"
  AURA_LOG_LEVEL: "info"
  AURA_COST_CAP_USD: "100.0"
```

### Secret

```bash
kubectl create secret generic aura-secrets \
  --from-literal=AGENT_API_TOKEN=$(openssl rand -hex 32) \
  --from-literal=OPENROUTER_API_KEY=$OPENROUTER_API_KEY
```

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aura-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: aura-api
  template:
    metadata:
      labels:
        app: aura-api
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: api
        image: ghcr.io/asshat1981ar/aura-cli:1.0.0
        ports:
        - containerPort: 8001
        envFrom:
        - configMapRef:
            name: aura-config
        - secretRef:
            name: aura-secrets
        resources:
          limits:
            memory: "512Mi"
            cpu: "1000m"
          requests:
            memory: "256Mi"
            cpu: "500m"
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: false  # SQLite needs writes
          capabilities:
            drop:
            - ALL
        volumeMounts:
        - name: memory
          mountPath: /app/memory
        - name: logs
          mountPath: /app/logs
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8001
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: memory
        emptyDir: {}
      - name: logs
        emptyDir: {}
```

### Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: aura-api
spec:
  selector:
    app: aura-api
  ports:
  - port: 80
    targetPort: 8001
  type: ClusterIP
```

### Ingress (with TLS)

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: aura-ingress
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - aura.example.com
    secretName: aura-tls
  rules:
  - host: aura.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: aura-api
            port:
              number: 80
```

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AGENT_API_TOKEN` | Yes | — | API authentication token |
| `OPENROUTER_API_KEY` | Yes* | — | OpenRouter API key |
| `OPENAI_API_KEY` | Yes* | — | OpenAI API key (alternative) |
| `AURA_ENV` | No | `development` | Environment name |
| `AURA_LOG_LEVEL` | No | `info` | Log level (debug/info/warn/error) |
| `AURA_COST_CAP_USD` | No | — | Monthly cost limit |
| `AURA_ENABLE_SWARM` | No | `0` | Enable agent swarm mode |
| `REDIS_URL` | No | — | Redis connection URL |

*At least one LLM provider key required

### Cost Control

Set a monthly cost cap to prevent runaway spending:

```bash
export AURA_COST_CAP_USD=50.00  # $50/month limit
```

When the cap is reached, the system will:
1. Log an error event
2. Raise `CostCapExceededError`
3. Reject subsequent model calls

---

## Security Hardening

### Network Security

1. **Firewall Rules**
   ```bash
   # Allow only necessary inbound traffic
   ufw default deny incoming
   ufw allow 22/tcp   # SSH (restrict to bastion)
   ufw allow 80/tcp   # HTTP (if not behind LB)
   ufw allow 443/tcp  # HTTPS
   ufw enable
   ```

2. **API Token Security**
   - Generate strong tokens: `openssl rand -hex 32`
   - Rotate tokens quarterly
   - Store in secrets manager (not env files)

3. **Sandbox Security**
   - Review `docs/security/sandbox-audit-v1.0.md`
   - Monitor `aura_sandbox_violations_total` metric
   - Enable violation alerting

### Data Protection

1. **Encryption at Rest**
   - SQLite: Use SQLCipher for encrypted DB
   - Redis: Enable TLS + AUTH
   - Backups: Encrypt before upload

2. **Encryption in Transit**
   - TLS 1.3 for all external traffic
   - mTLS for internal service mesh (optional)

### Access Control

```bash
# Create read-only user for monitoring
kubectl create rolebinding aura-readonly \
  --clusterrole=view \
  --serviceaccount=aura:default \
  --namespace=aura
```

---

## Monitoring & Observability

### Prometheus Metrics

Key metrics to monitor:

| Metric | Alert Threshold | Description |
|--------|-----------------|-------------|
| `aura_pipeline_runs_total` | — | Total pipeline runs |
| `aura_active_pipeline_runs` | >10 | Concurrent executions |
| `aura_goal_queue_depth` | >100 | Queue backlog |
| `aura_sandbox_violations_total` | >0 | Security violations |
| `http_request_duration_seconds` | p99>5s | API latency |

### Grafana Dashboard

Import the dashboard from `docker/grafana/dashboards/aura-dashboard.json`

### Alerting Rules

```yaml
# Example PrometheusRule
groups:
- name: aura-alerts
  rules:
  - alert: AuraHighErrorRate
    expr: rate(aura_pipeline_runs_total{status="failure"}[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High AURA pipeline failure rate"
  
  - alert: AuraSandboxViolation
    expr: increase(aura_sandbox_violations_total[1h]) > 0
    labels:
      severity: critical
    annotations:
      summary: "Sandbox security violation detected"
```

### Log Aggregation

Structured JSON logs are written to stderr. Configure your log aggregator:

```bash
# Fluent Bit example
[PARSER]
    Name        aura
    Format      json
    Time_Key    ts
    Time_Format %Y-%m-%dT%H:%M:%S.%f%z
```

---

## Backup & Recovery

### What to Backup

| Data | Location | Frequency | Retention |
|------|----------|-----------|-----------|
| Brain DB | `memory/brain_v2.db` | Daily | 30 days |
| Goal Queue | `memory/goal_queue.json` | Hourly | 7 days |
| Logs | `logs/` | Stream to S3 | 90 days |
| Telemetry | `telemetry.db` | Daily | 30 days |

### Backup Script

```bash
#!/bin/bash
# backup-aura.sh

BACKUP_DIR="/backups/aura/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup SQLite databases
cp memory/brain_v2.db "$BACKUP_DIR/"
cp telemetry.db "$BACKUP_DIR/"

# Backup JSON files
cp memory/goal_queue.json "$BACKUP_DIR/"
cp memory/skill_weights.json "$BACKUP_DIR/"

# Compress and upload
tar czf "$BACKUP_DIR.tar.gz" -C "$BACKUP_DIR" .
aws s3 cp "$BACKUP_DIR.tar.gz" s3://aura-backups/
rm -rf "$BACKUP_DIR" "$BACKUP_DIR.tar.gz"
```

### Recovery Procedure

1. Stop AURA services
2. Restore from backup: `cp backup/brain_v2.db memory/`
3. Restart services
4. Verify health: `curl /health`

---

## Troubleshooting

### Common Issues

#### High Memory Usage

```bash
# Check memory usage
kubectl top pods -l app=aura-api

# Enable memory profiling
export AURA_LOG_LEVEL=debug
```

#### Sandbox Violations

```bash
# View recent violations
kubectl logs deployment/aura-api | grep sandbox_violation

# Check violation details in telemetry
sqlite3 telemetry.db "SELECT * FROM telemetry WHERE event='sandbox_violation_attempt'"
```

#### Queue Backlog

```bash
# Check queue depth
curl -H "Authorization: Bearer $TOKEN" http://aura-api/webhook/status/queue

# Clear stuck goals
python3 -c "from core.goal_queue import GoalQueue; q = GoalQueue(); q.clear()"
```

### Debug Mode

Enable debug logging:

```bash
export AURA_LOG_LEVEL=debug
export AURA_DEBUG=1
```

### Support

- Issues: https://github.com/asshat1981ar/aura-cli/issues
- Security: security@aura-cli.dev
- Documentation: https://docs.aura-cli.dev

---

## Appendix A: Security Checklist

- [ ] API token is strong (32+ random characters)
- [ ] TLS enabled for all endpoints
- [ ] Redis password set (if used)
- [ ] Cost cap configured
- [ ] Sandbox violations monitored
- [ ] Backups encrypted
- [ ] Log retention configured
- [ ] Resource limits set
- [ ] Health checks configured
- [ ] Alerting rules active

---

## Appendix B: Migration Guide

### From v0.1.0 to v1.0.0

1. Backup existing data
2. Update image tag to `1.0.0`
3. Add new required env vars:
   - `AURA_COST_CAP_USD` (optional but recommended)
4. Update liveness probe to use `/ready`
5. Deploy and verify
