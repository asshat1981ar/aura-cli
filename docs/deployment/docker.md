# Docker Deployment

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/asshat1981ar/aura-cli.git
cd aura-cli

# Start all services
docker-compose -f infra/docker/docker-compose.prod.yml up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Services Included

- **AURA API**: Main API server (port 8000)
- **Redis**: Caching and message broker (port 6379)
- **PostgreSQL**: Persistent storage (port 5432)
- **Prometheus**: Metrics collection (port 9090)
- **Grafana**: Dashboards and visualization (port 3000)

## Building the Image

### Production Build

```bash
docker build -f infra/docker/Dockerfile.production -t aura-cli:latest .
```

### Running the Container

```bash
docker run -d \
  --name aura-api \
  -p 8000:8000 \
  -e AURA_ENV=production \
  -e REDIS_URL=redis://redis:6379/0 \
  -e DATABASE_URL=postgresql://aura:password@postgres:5432/aura \
  -v $(pwd)/memory:/app/memory \
  -v $(pwd)/logs:/app/logs \
  aura-cli:latest
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AURA_ENV` | Environment (development/production) | `production` |
| `AURA_HOST` | API server host | `0.0.0.0` |
| `AURA_PORT` | API server port | `8000` |
| `AURA_WORKERS` | Number of worker processes | `1` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `DATABASE_URL` | PostgreSQL connection URL | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `GITHUB_WEBHOOK_SECRET` | GitHub webhook secret | - |

## Docker Compose Configuration

### Basic Configuration

```yaml
version: '3.8'

services:
  aura-api:
    image: aura-cli:latest
    ports:
      - "8000:8000"
    environment:
      - AURA_ENV=production
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    volumes:
      - ./memory:/app/memory
      - ./logs:/app/logs

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data

volumes:
  redis-data:
```

### Production Configuration

See `infra/docker/docker-compose.prod.yml` for the complete production setup including:
- Multi-service orchestration
- Persistent volumes
- Health checks
- Resource limits
- Monitoring stack

## Entrypoint Commands

The Docker image supports different entrypoint commands:

```bash
# Run API server (default)
docker run aura-cli:latest

# Run background worker
docker run aura-cli:latest worker

# Run scheduler
docker run aura-cli:latest scheduler

# Run CLI command
docker run aura-cli:latest cli goal add "Fix bug"

# Run database migrations
docker run aura-cli:latest migrate
```

## Health Checks

The container includes a health check:

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

Check container health:

```bash
docker ps
# or
docker inspect --format='{{.State.Health.Status}}' aura-api
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs aura-api

# Check environment
docker inspect aura-api | grep -A 20 Env
```

### Permission Issues

```bash
# Fix volume permissions
sudo chown -R 1000:1000 ./memory ./logs
```

### Network Issues

```bash
# Verify network connectivity
docker network ls
docker network inspect <network_name>
```

## Security Best Practices

1. **Use specific image tags** instead of `latest`
2. **Run as non-root user** (already configured)
3. **Use secrets management** for sensitive data
4. **Enable read-only root filesystem**
5. **Scan images for vulnerabilities**:

```bash
docker scan aura-cli:latest
# or with Trivy
trivy image aura-cli:latest
```

## Updating

```bash
# Pull latest image
docker pull aura-cli:latest

# Recreate containers
docker-compose up -d --force-recreate

# Clean up old images
docker image prune -f
```

## Backup and Restore

### Backup

```bash
# Backup memory and logs
tar czf aura-backup-$(date +%Y%m%d).tar.gz memory/ logs/

# Backup database
docker exec aura-postgres pg_dump -U aura aura > aura-db-$(date +%Y%m%d).sql
```

### Restore

```bash
# Restore memory and logs
tar xzf aura-backup-20260402.tar.gz

# Restore database
docker exec -i aura-postgres psql -U aura aura < aura-db-20260402.sql
```
