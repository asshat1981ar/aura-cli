# AURA Deployment Guide

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone the repository**
```bash
git clone https://github.com/your-org/aura-cli.git
cd aura-cli
```

2. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Deploy**
```bash
./scripts/deploy.sh production
```

## Deployment Options

### Option 1: Docker Compose (Single Server)

Best for small to medium deployments.

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Update deployment
docker-compose pull
docker-compose up -d
```

**Services:**
- `api`: AURA API server (port 8000)
- `web`: Web UI (port 80)
- `redis`: Caching (optional)
- `prometheus`: Metrics collection
- `grafana`: Monitoring dashboard

### Option 2: Manual Deployment

For custom infrastructure or cloud providers.

#### Prerequisites
- Python 3.11+
- Node.js 20+
- Nginx (for web UI)
- Redis (optional)

#### API Server
```bash
# Install dependencies
pip install -r requirements.txt

# Install AURA
pip install -e .

# Run server
python3 -m aura_cli.api_server
```

#### Web UI
```bash
cd web-ui
npm install
npm run build

# Serve with Nginx or any static file server
```

### Option 3: Cloud Deployment

#### AWS ECS/Fargate
```bash
# Build images
docker build -t aura-api .
docker build -f Dockerfile.web -t aura-web .

# Push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag aura-api:latest <account>.dkr.ecr.<region>.amazonaws.com/aura-api:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/aura-api:latest
```

#### Google Cloud Run
```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/aura-api
gcloud builds submit --tag gcr.io/PROJECT_ID/aura-web --file Dockerfile.web

# Deploy
gcloud run deploy aura-api --image gcr.io/PROJECT_ID/aura-api
gcloud run deploy aura-web --image gcr.io/PROJECT_ID/aura-web
```

#### Kubernetes
```bash
# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

## Configuration

### Environment Variables

See `.env.example` for all available options.

**Required:**
- `AURA_SECRET_KEY`: Secret for encryption
- `AURA_JWT_SECRET`: JWT signing key

**Optional:**
- `REDIS_URL`: Redis connection for caching
- `SENTRY_DSN`: Error tracking
- `GITHUB_TOKEN`: GitHub integration

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name aura.example.com;
    
    location / {
        proxy_pass http://localhost:80;
    }
    
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    location /ws/ {
        proxy_pass http://localhost:8000/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### SSL/TLS (Let's Encrypt)

```bash
# Install certbot
apt-get install certbot python3-certbot-nginx

# Obtain certificate
certbot --nginx -d aura.example.com

# Auto-renewal
certbot renew --dry-run
```

## Health Checks

The API provides health endpoints:

- `GET /api/health` - System health status
- `GET /api/stats` - System statistics

**Docker Health Check:**
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1
```

## Monitoring

### Prometheus Metrics

Available at `http://localhost:9090`

Key metrics:
- `aura_goals_total`: Total goals processed
- `aura_agents_active`: Active agents
- `aura_api_requests`: API request count
- `aura_api_latency`: Request latency

### Grafana Dashboards

Access at `http://localhost:3001`

Default dashboards:
- System Overview
- API Performance
- Agent Metrics
- Goal Queue Status

### Logging

View logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api

# Web UI
docker-compose logs -f web
```

Log format: JSON structured logging

## Backup and Recovery

### Data Backup

```bash
# Backup SQLite database
docker exec aura-api tar czf /tmp/backup.tar.gz /data
docker cp aura-api:/tmp/backup.tar.gz ./backup-$(date +%Y%m%d).tar.gz
```

### Restore

```bash
# Restore from backup
docker cp backup-20240101.tar.gz aura-api:/tmp/
docker exec aura-api tar xzf /tmp/backup-20240101.tar.gz -C /
```

## Scaling

### Horizontal Scaling

Run multiple API instances behind a load balancer:

```yaml
# docker-compose.scale.yml
services:
  api:
    deploy:
      replicas: 3
```

### Vertical Scaling

Adjust resource limits:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
```

## Troubleshooting

### Common Issues

**Services won't start:**
```bash
# Check logs
docker-compose logs

# Verify ports are available
netstat -tlnp | grep -E '8000|80'
```

**Web UI not loading:**
- Check API server is running
- Verify WebSocket connection
- Clear browser cache

**High memory usage:**
- Restart services: `docker-compose restart`
- Check for memory leaks in logs
- Scale up resources

### Debug Mode

```bash
# Run in foreground
docker-compose up

# Enable debug logging
AURA_LOG_LEVEL=debug docker-compose up
```

## Security Checklist

- [ ] Change default secrets in `.env`
- [ ] Enable HTTPS with valid SSL certificate
- [ ] Configure firewall rules
- [ ] Set up fail2ban for brute force protection
- [ ] Regular security updates
- [ ] Backup encryption
- [ ] Access logging enabled
- [ ] Rate limiting configured

## Performance Tuning

### Database

SQLite (default):
- Good for small deployments
- Enable WAL mode for better concurrency

PostgreSQL (optional):
```bash
# Use PostgreSQL for larger deployments
pip install psycopg2-binary
# Set AURA_DB_URL=postgresql://user:pass@localhost/aura
```

### Caching

Enable Redis:
```bash
# Start Redis
docker-compose up -d redis

# Configure in .env
REDIS_ENABLED=true
REDIS_URL=redis://redis:6379/0
```

### CDN (Production)

Serve static assets from CDN:
```bash
# Build with CDN path
CDN_URL=https://cdn.example.com npm run build
```

## Updates

### Rolling Updates

```bash
# Pull latest images
docker-compose pull

# Restart with zero downtime
docker-compose up -d --no-deps --scale api=2 api
docker-compose up -d --scale api=1 api
```

### Database Migrations

```bash
# Run migrations
python3 -m aura_cli migrate
```

## Support

- Documentation: `docs/`
- Issues: GitHub Issues
- Discussions: GitHub Discussions
