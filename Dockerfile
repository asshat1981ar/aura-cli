# ---- Build stage ----
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml requirements.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir --upgrade pip setuptools wheel

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY aura_cli/ aura_cli/
COPY core/ core/
COPY agents/ agents/
COPY memory/ memory/
COPY tools/ tools/
COPY orchestrator_hub/ orchestrator_hub/
COPY main.py run_aura.sh ./

RUN pip install --no-cache-dir --prefix=/install .

# ---- Runtime stage ----
FROM python:3.12-slim

LABEL maintainer="aura-cli"
LABEL description="AURA CLI — Autonomous software development platform"

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
WORKDIR /app
COPY --from=builder /build /app

# Create non-root user
RUN groupadd -r aura && useradd -r -g aura -d /app aura \
    && mkdir -p /app/memory /app/logs \
    && chown -R aura:aura /app

USER aura

# Runtime config
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV AURA_SKIP_CHDIR=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default: MCP server mode. Override with: docker run aura-cli aura goal run --dry-run
CMD ["uvicorn", "tools.mcp_server:app", "--host", "0.0.0.0", "--port", "8000"]
