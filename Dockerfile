# AURA API Server - Production Dockerfile (multi-stage build)
# Updated for Redis TLS and ReAct Orchestration

# Stage 1: Builder
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    musl-dev \
    libffi-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt pyproject.toml ./

# Install dependencies
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Copy source
COPY . .
# Note: In this project, code is in root directories like core/, agents/
RUN pip install --no-cache-dir --prefix=/install -e . --no-deps

# Stage 2: Runtime
FROM python:3.12-slim

# No build tools in the runtime image
ENV PYTHONPATH=/install/lib/python3.12/site-packages \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    redis-tools \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /install

# Create non-root user with fixed UID for container security
RUN useradd -m -u 1000 aura

# Copy application code with correct ownership
# Flat structure: copying everything
COPY --chown=aura:aura . .

# Create data directory and set permissions
RUN mkdir -p /data && chown -R aura:aura /data /app

# Switch to non-root user
USER aura

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Entry point per pyproject.toml
CMD ["aura", "goal", "run"]
