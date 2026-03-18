# syntax=docker/dockerfile:1

# --- Stage 1: builder -------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

WORKDIR /build
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: runtime -------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# Bring in installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Clean bytecode caches
RUN find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true

# Run as non-root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "tools.mcp_server:app", "--host", "0.0.0.0", "--port", "8000"]
