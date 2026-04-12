# Log Aggregation Integration Guide — AURA

> **Audience:** Platform / SRE engineers  
> **Updated:** Sprint 8  
> **Related:** `core/logging_utils.py`, `docker-compose.yml`

---

## Table of Contents

1. [AURA Log Format](#aura-log-format)
2. [Fluentd / Fluent Bit Sidecar](#fluentd--fluent-bit-sidecar)
3. [Loki + Promtail Alternative](#loki--promtail-alternative)
4. [LogQL Queries for Common Ops Tasks](#logql-queries)
5. [Structured Log Fields Reference](#structured-log-fields-reference)

---

## AURA Log Format

AURA emits structured **JSON logs** on `stdout` via `core/logging_utils.log_json`.

Each log line is a single-line JSON object with the following base fields:

```json
{
  "timestamp": "2025-07-14T12:34:56.789Z",
  "level": "INFO",
  "event": "aura_webhook_goal_received",
  "details": {
    "goal_id": "abc-123",
    "goal": "Analyse PR #42",
    "priority": 5
  },
  "service": "aura-api",
  "version": "1.0.0"
}
```

**Level values:** `DEBUG` · `INFO` · `WARN` · `ERROR`

To tail raw structured logs locally:
```bash
docker logs aura-api -f 2>&1 | python3 -m json.tool
```

---

## Fluentd / Fluent Bit Sidecar

Fluent Bit is the preferred lightweight forwarder. Add it as a sidecar/companion container that reads Docker container logs and ships them to an aggregation backend (Elasticsearch, OpenSearch, Loki, CloudWatch, etc.).

### docker-compose.yml addition

```yaml
# Add to docker-compose.yml or docker-compose.prod.yml

services:
  # ... existing services ...

  fluent-bit:
    image: fluent/fluent-bit:3
    container_name: aura-fluent-bit
    restart: unless-stopped
    volumes:
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./docker/fluent-bit/fluent-bit.conf:/fluent-bit/etc/fluent-bit.conf:ro
      - ./docker/fluent-bit/parsers.conf:/fluent-bit/etc/parsers.conf:ro
    networks:
      - aura-network
    depends_on:
      - api
    environment:
      - FLB_LOG_LEVEL=warn
```

### `docker/fluent-bit/fluent-bit.conf`

```ini
[SERVICE]
    Flush        5
    Log_Level    warn
    Daemon       Off
    Parsers_File parsers.conf
    HTTP_Server  On
    HTTP_Listen  0.0.0.0
    HTTP_Port    2020

# ── INPUT: Docker container logs ─────────────────────────────────────────────
[INPUT]
    Name              tail
    Path              /var/lib/docker/containers/*/*.log
    Parser            docker
    Tag               docker.<container_id>
    Refresh_Interval  5
    Mem_Buf_Limit     50MB
    Skip_Long_Lines   On
    DB                /tmp/fluent-bit-docker.db

# ── FILTER: Add metadata and parse AURA JSON payloads ────────────────────────
[FILTER]
    Name         record_modifier
    Match        docker.*
    Record       host ${HOSTNAME}
    Record       env  production

[FILTER]
    Name         grep
    Match        docker.*
    Regex        source (aura-api|aura-web|aura-worker)

[FILTER]
    Name         parser
    Match        docker.*
    Key_Name     log
    Parser       aura_json
    Reserve_Data True

# ── OUTPUT: Loki ──────────────────────────────────────────────────────────────
[OUTPUT]
    Name            loki
    Match           docker.*
    Host            loki
    Port            3100
    Labels          job=aura, container=$container_name, level=$level
    Label_Keys      $event,$service
    Line_Format     json
    Auto_Kubernetes_Labels Off

# ── OUTPUT: Elasticsearch (alternative — comment out Loki block above) ────────
# [OUTPUT]
#     Name            es
#     Match           docker.*
#     Host            elasticsearch
#     Port            9200
#     Index           aura-logs-%Y.%m.%d
#     Logstash_Format On
#     Include_Tag_Key On
```

### `docker/fluent-bit/parsers.conf`

```ini
[PARSER]
    Name        docker
    Format      json
    Time_Key    time
    Time_Format %Y-%m-%dT%H:%M:%S.%L%z

[PARSER]
    Name        aura_json
    Format      json
    Time_Key    timestamp
    Time_Format %Y-%m-%dT%H:%M:%S.%LZ
```

---

## Loki + Promtail Alternative

[Grafana Loki](https://grafana.com/oss/loki/) + Promtail is the recommended stack when Grafana is already deployed (as in `docker-compose.yml`).

### docker-compose addition

```yaml
services:
  loki:
    image: grafana/loki:2.9.0
    container_name: aura-loki
    restart: unless-stopped
    ports:
      - "3100:3100"
    volumes:
      - ./docker/loki/loki-config.yml:/etc/loki/local-config.yaml:ro
      - loki-data:/loki
    networks:
      - aura-network
    command: -config.file=/etc/loki/local-config.yaml

  promtail:
    image: grafana/promtail:2.9.0
    container_name: aura-promtail
    restart: unless-stopped
    volumes:
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./docker/promtail/promtail-config.yml:/etc/promtail/config.yml:ro
    networks:
      - aura-network
    command: -config.file=/etc/promtail/config.yml
    depends_on:
      - loki

volumes:
  loki-data:
```

### `docker/loki/loki-config.yml`

```yaml
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9096

common:
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    instance_addr: 127.0.0.1
    kvstore:
      store: inmemory

schema_config:
  configs:
    - from: 2024-01-01
      store: boltdb-shipper
      object_store: filesystem
      schema: v12
      index:
        prefix: index_
        period: 24h

ruler:
  alertmanager_url: http://localhost:9093

limits_config:
  retention_period: 744h    # 31 days
```

### `docker/promtail/promtail-config.yml`

```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: docker-containers
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
        filters:
          - name: name
            values: [aura-api, aura-web, aura-worker]
    relabel_configs:
      - source_labels: [__meta_docker_container_name]
        target_label: container
      - source_labels: [__meta_docker_container_log_stream]
        target_label: stream
    pipeline_stages:
      # Parse AURA structured JSON logs
      - json:
          expressions:
            level:     level
            event:     event
            service:   service
            timestamp: timestamp
      # Promote `level` to a Loki label for fast filtering
      - labels:
          level:
          event:
          service:
      - timestamp:
          source: timestamp
          format: RFC3339Nano
```

### Add Loki datasource to Grafana

Extend `infra/grafana/datasources.yml`:

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true

  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    jsonData:
      maxLines: 1000
```

---

## LogQL Queries

Use these in the Grafana **Explore** view with the Loki datasource.

### View all ERROR logs from AURA API

```logql
{container="aura-api"} | json | level="ERROR"
```

### Show recent goal received events

```logql
{container="aura-api"} | json
  | event="aura_webhook_goal_received"
  | line_format "{{.timestamp}} goal_id={{.details_goal_id}} goal={{.details_goal}}"
```

### Count errors per minute (rate graph)

```logql
sum(rate({container="aura-api"} | json | level="ERROR" [1m]))
```

### Find all failed pipeline runs in the last hour

```logql
{container="aura-api"} | json
  | event=~"aura_webhook_goal_failed|aura_pipeline.*failed"
  | line_format "{{.timestamp}} {{.details_goal_id}} {{.details_error}}"
```

### Latency outliers — requests taking >5 s

```logql
{container="aura-api"} | json
  | details_duration_ms > 5000
  | line_format "{{.timestamp}} endpoint={{.details_endpoint}} duration={{.details_duration_ms}}ms"
```

### Show sandbox violations

```logql
{container="aura-api"} | json
  | event="aura_sandbox_violation"
  | line_format "{{.timestamp}} type={{.details_type}} cmd={{.details_command}}"
```

### Tail all WARN+ logs across all AURA containers

```logql
{job="aura"} | json | level=~"WARN|ERROR"
```

### Goal queue depth over time (parsed from logs)

```logql
sum_over_time(
  {container="aura-api"} | json
    | event="aura_queue_depth_snapshot"
    | unwrap details_depth [1m]
)
```

---

## Structured Log Fields Reference

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | ISO-8601 string | UTC timestamp of the event |
| `level` | string | `DEBUG` / `INFO` / `WARN` / `ERROR` |
| `event` | string | Machine-readable event name (snake_case) |
| `service` | string | Service name (e.g., `aura-api`) |
| `version` | string | Application version |
| `details.*` | object | Event-specific payload (goal_id, error, duration_ms, …) |

### Common Event Names

| Event | Level | Description |
|-------|-------|-------------|
| `aura_webhook_goal_received` | INFO | A new goal was submitted via webhook |
| `aura_webhook_goal_started` | INFO | Goal execution began |
| `aura_webhook_goal_completed` | INFO | Goal completed successfully |
| `aura_webhook_goal_failed` | WARN | Goal execution failed |
| `aura_pipeline_run_started` | INFO | Pipeline run initiated |
| `aura_pipeline_run_finished` | INFO | Pipeline run completed |
| `aura_sandbox_violation` | WARN | Denylist command blocked |
| `aura_run_tool_timeout` | WARN | Tool execution timed out |
| `aura_server_startup` | INFO | API server started |
| `aura_runtime_init_error` | ERROR | Runtime initialisation failed |
