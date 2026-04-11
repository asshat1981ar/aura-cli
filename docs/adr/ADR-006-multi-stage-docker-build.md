# ADR-006: Multi-Stage Docker Build

**Date:** 2026-04-10
**Status:** Accepted
**Deciders:** Copilot (Sprint 4 automation)

---

## Context

The original `Dockerfile` was a single-stage build that:
- Installed `gcc` (a build-time dependency) into the production image
- Copied the entire repository including test files, scripts, and memory data
- Used the outdated entry point `python3 -m aura_cli.api_server`
- Had no `.dockerignore`, so dev artifacts (`.git`, `__pycache__`, `.venv`) were included
- Exposed port 8001 inconsistently with the actual server port (8000)

This produced unnecessarily large images and included build toolchains not needed at runtime.

---

## Decision

Switch to a **two-stage Docker build**:

### Stage 1: `builder`
```dockerfile
FROM python:3.11-slim AS builder
RUN apt-get install -y gcc
COPY requirements.txt pyproject.toml .
RUN pip install --prefix=/install -r requirements.txt
RUN pip install --prefix=/install -e . --no-deps
```

### Stage 2: `runtime`
```dockerfile
FROM python:3.11-slim
COPY --from=builder /install /install
ENV PYTHONPATH=/install/lib/python3.11/site-packages
# no gcc, no build tools
COPY --chown=aura:aura . .
CMD ["uvicorn", "aura_cli.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Additionally:
- Created `.dockerignore` (79 lines) to exclude `.git`, test artifacts, `memory/` runtime data, `web-ui/node_modules`
- Updated CMD to `uvicorn aura_cli.server:app` (the correct production server)
- Aligned port to `8000` across Dockerfile, docker-compose.yml, and docker-compose.prod.yml

---

## Consequences

**Positive:**
- Production image contains no C compiler or build tools → smaller attack surface
- `.dockerignore` prevents dev artifacts from entering the build context → faster builds, smaller layers
- Consistent port (8000) removes confusion between MCP port (8001) and API server port
- Correct entry point (`uvicorn aura_cli.server:app`) replaces stale module reference

**Negative / Trade-offs:**
- `pip install -e .` in the builder stage installs in editable mode; the runtime stage gets the wheel-installed version — this means `importlib.metadata` package discovery works correctly but the source tree is still present in the runtime image via `COPY . .`
- `PYTHONPATH` must be set explicitly to point at the `/install` prefix; if Python version changes from 3.11, the path needs updating

**Follow-on:**
- Consider pinning `python:3.11-slim` to a digest hash for supply-chain reproducibility
- Evaluate replacing `pip install -e .` with a proper wheel build (`python -m build`) in the builder stage
