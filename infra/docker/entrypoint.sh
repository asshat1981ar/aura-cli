#!/bin/bash
set -e

# AURA CLI Production Entrypoint
# Usage: entrypoint.sh [server|worker|cli|migrate]

ROLE="${1:-server}"
export AURA_ROLE="$ROLE"

case "$ROLE" in
    server)
        echo "Starting AURA MCP Server..."
        exec uvicorn tools.mcp_server:app \
            --host "${AURA_HOST:-0.0.0.0}" \
            --port "${AURA_PORT:-8000}" \
            --workers "${AURA_WORKERS:-1}" \
            --loop uvloop \
            --http h11 \
            --lifespan on
        ;;
    
    worker)
        echo "Starting AURA Background Worker..."
        exec python -m core.task_worker \
            --queue "${AURA_QUEUE:-default}" \
            --concurrency "${AURA_WORKER_CONCURRENCY:-4}"
        ;;
    
    scheduler)
        echo "Starting AURA Task Scheduler..."
        exec python -m core.task_scheduler
        ;;
    
    cli)
        shift
        exec python main.py "$@"
        ;;
    
    migrate)
        echo "Running database migrations..."
        exec python -m core.migrations upgrade head
        ;;
    
    healthcheck)
        # Simple health check for Docker/K8s
        python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" 2>/dev/null || exit 1
        ;;
    
    *)
        echo "Unknown role: $ROLE"
        echo "Usage: $0 [server|worker|scheduler|cli|migrate|healthcheck]"
        exit 1
        ;;
esac
