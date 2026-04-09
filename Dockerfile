# AURA API Server - Production Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with fixed UID for container security
RUN useradd -m -u 1000 aura

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code with correct ownership
COPY --chown=aura:aura . .

# Install the package
RUN pip install -e .

# Create data directory and set permissions
RUN mkdir -p /data && chown -R aura:aura /data /app

# Switch to non-root user
USER aura

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose port
EXPOSE 8001

# Run the API server
CMD ["python3", "-m", "aura_cli.api_server"]
