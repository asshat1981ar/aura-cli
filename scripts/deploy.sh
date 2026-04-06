#!/bin/bash
# AURA Deployment Script
# Usage: ./scripts/deploy.sh [environment]
# Environments: dev, staging, production

set -e

ENVIRONMENT=${1:-dev}
VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "dev")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== AURA Deployment Script ===${NC}"
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo ""

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|production)$ ]]; then
    echo -e "${RED}Error: Invalid environment. Use: dev, staging, or production${NC}"
    exit 1
fi

# Build Web UI
echo -e "${YELLOW}Building Web UI...${NC}"
cd web-ui
npm ci
npm run build
cd ..

# Build Docker images
echo -e "${YELLOW}Building Docker images...${NC}"
docker build -t aura-api:$VERSION .
docker build -f Dockerfile.web -t aura-web:$VERSION .

# Tag images
docker tag aura-api:$VERSION aura-api:$ENVIRONMENT
docker tag aura-web:$VERSION aura-web:$ENVIRONMENT

# Deploy based on environment
case $ENVIRONMENT in
    dev)
        echo -e "${YELLOW}Deploying to development...${NC}"
        docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
        ;;
    staging)
        echo -e "${YELLOW}Deploying to staging...${NC}"
        docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d
        ;;
    production)
        echo -e "${YELLOW}Deploying to production...${NC}"
        echo -e "${RED}WARNING: This will deploy to production!${NC}"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            echo "Deployment cancelled"
            exit 0
        fi
        docker-compose up -d
        ;;
esac

# Wait for services to be healthy
echo -e "${YELLOW}Waiting for services to be healthy...${NC}"
sleep 10

# Health check
echo -e "${YELLOW}Running health checks...${NC}"
if curl -sf http://localhost:8000/api/health > /dev/null; then
    echo -e "${GREEN}✓ API server is healthy${NC}"
else
    echo -e "${RED}✗ API server health check failed${NC}"
    exit 1
fi

if curl -sf http://localhost:80 > /dev/null; then
    echo -e "${GREEN}✓ Web server is healthy${NC}"
else
    echo -e "${RED}✗ Web server health check failed${NC}"
    exit 1
fi

# Display deployment info
echo ""
echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo "Version: $VERSION"
echo "Environment: $ENVIRONMENT"
echo ""
echo "URLs:"
echo "  Web UI: http://localhost"
echo "  API: http://localhost:8000"
echo "  Health: http://localhost:8000/api/health"

if [ "$ENVIRONMENT" == "production" ]; then
    echo ""
    echo -e "${YELLOW}Remember to:${NC}"
    echo "  - Monitor logs: docker-compose logs -f"
    echo "  - Check metrics: http://localhost:3001 (Grafana)"
    echo "  - Review alerts in Prometheus"
fi
