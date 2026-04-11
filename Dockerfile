# AURA API Server - Production Dockerfile (multi-stage build)

# ── builder ───────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt pyproject.toml ./

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Copy source so the editable install can resolve the package
COPY . .
RUN pip install --no-cache-dir --prefix=/install -e . --no-deps

# ── runtime ───────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# No build tools in the runtime image

ENV PYTHONPATH=/install/lib/python3.11/site-packages \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /install

# Create non-root user with fixed UID for container security
RUN useradd -m -u 1000 aura

# Copy application code with correct ownership
COPY --chown=aura:aura . .

# Create data directory and set permissions
RUN mkdir -p /data && chown -R aura:aura /data /app

# Install curl for the health check (minimal, no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER aura

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

CMD ["uvicorn", "aura_cli.server:app", "--host", "0.0.0.0", "--port", "8000"]
