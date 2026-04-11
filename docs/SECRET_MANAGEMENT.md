# Secret Management Guide — AURA

> **Audience:** DevOps engineers, developers deploying AURA  
> **Last updated:** Sprint 8  
> **Security policy:** See [SECURITY.md](SECURITY.md)

---

## Table of Contents

1. [Secrets Inventory](#secrets-inventory)
2. [What Never Goes in Git](#what-never-goes-in-git)
3. [Docker Secrets Pattern (Swarm)](#docker-secrets-pattern-swarm)
4. [Environment Variable Injection Pattern (Compose / K8s)](#environment-variable-injection-pattern)
5. [Rotation Procedures](#rotation-procedures)
   - [AURA_API_KEY](#aura_api_key-rotation)
   - [AGENT_API_TOKEN](#agent_api_token-rotation)
6. [Secret Scanning](#secret-scanning)

---

## Secrets Inventory

| Secret | Purpose | Owner | Rotation Period |
|--------|---------|-------|-----------------|
| `AURA_API_KEY` | Authenticates HTTP requests to the AURA API server | Platform team | 90 days |
| `AGENT_API_TOKEN` | Token used by n8n / external agents to call AURA endpoints | Platform team | 90 days |
| `ANTHROPIC_API_KEY` | Anthropic Claude API access | AI team | Per provider policy |
| `OPENAI_API_KEY` | OpenAI API access (optional adapter) | AI team | Per provider policy |
| `N8N_WEBHOOK_URL` | n8n webhook URL for goal dispatch | Automation team | On compromise |
| `GF_SECURITY_ADMIN_PASSWORD` | Grafana admin password | Platform team | 180 days |
| `REDIS_PASSWORD` | Redis AUTH password (if enabled) | Platform team | 180 days |
| `DB_PASSWORD` | Database password (future PostgreSQL migration) | Platform team | 90 days |
| `GITHUB_TOKEN` | GitHub App / PAT for repository access | Platform team | 30 days |

---

## What Never Goes in Git

The following **must never** be committed to the repository.  
They are covered by `.gitignore`:

```
# .gitignore (relevant entries)
.env
.env.*
!.env.example
*.key
*.pem
*.p12
secrets/
docker/ssl/
aura_auth.db
```

### Additional Safeguards

- **Pre-commit hook** — install [`detect-secrets`](https://github.com/Yelp/detect-secrets):
  ```bash
  pip install detect-secrets
  detect-secrets scan > .secrets.baseline
  # Add to .pre-commit-config.yaml:
  #   - repo: https://github.com/Yelp/detect-secrets
  #     hooks: [id: detect-secrets, args: ['--baseline', '.secrets.baseline']]
  ```

- **GitHub secret scanning** is enabled on the repository. Push protection blocks secrets matching known patterns.

- **Emergency:** If a secret is committed, rotate it immediately (see [Rotation Procedures](#rotation-procedures)) **and** rewrite git history:
  ```bash
  git filter-repo --path <file-with-secret> --invert-paths
  # OR
  gh secret delete <SECRET_NAME>
  ```

---

## Docker Secrets Pattern (Swarm)

For Docker Swarm deployments, use native Docker secrets to avoid passing sensitive values as plain environment variables.

### Step 1 — Create Secrets

```bash
# From a file:
echo -n "super-secret-api-key" | docker secret create aura_api_key -

# From environment:
printf '%s' "$AURA_API_KEY" | docker secret create aura_api_key -

# List secrets:
docker secret ls
```

### Step 2 — Reference in Compose / Stack File

```yaml
# docker-compose.prod.yml (Swarm stack)
version: '3.8'

services:
  api:
    image: ghcr.io/asshat1981ar/aura-cli:${VERSION:-latest}
    secrets:
      - aura_api_key
      - agent_api_token
      - anthropic_api_key
    environment:
      # Point the app at the mounted secret file paths:
      - AURA_API_KEY_FILE=/run/secrets/aura_api_key
      - AGENT_API_TOKEN_FILE=/run/secrets/agent_api_token
      - ANTHROPIC_API_KEY_FILE=/run/secrets/anthropic_api_key

secrets:
  aura_api_key:
    external: true          # must be created via `docker secret create` above
  agent_api_token:
    external: true
  anthropic_api_key:
    external: true
```

### Step 3 — Read Secrets at Runtime

In Python, read from the mounted file path at startup:

```python
import os, pathlib

def _read_secret(env_var: str, file_env_var: str | None = None) -> str | None:
    """Read a secret from env var or from a Docker secrets file."""
    if file_env_var:
        path = os.getenv(file_env_var)
        if path:
            try:
                return pathlib.Path(path).read_text().strip()
            except OSError:
                pass
    return os.getenv(env_var)

AURA_API_KEY = _read_secret("AURA_API_KEY", "AURA_API_KEY_FILE")
```

> **Note:** Docker secrets files are mounted at `/run/secrets/<name>` inside the container and are only accessible to the service that declares them.

---

## Environment Variable Injection Pattern

For **non-Swarm** Docker Compose or Kubernetes, inject secrets via environment variables sourced from a `.env` file or a secrets manager.

### Compose with `.env` file

```bash
# .env (never committed — in .gitignore)
AURA_API_KEY=<secret>
AGENT_API_TOKEN=<secret>
ANTHROPIC_API_KEY=<secret>
GF_SECURITY_ADMIN_PASSWORD=<secret>
```

```yaml
# docker-compose.prod.yml
services:
  api:
    env_file:
      - .env        # injected at `docker compose up` time
    environment:
      - AURA_ENV=production
```

**Deployment workflow:**
```bash
# On the production host (secret injected from CI/CD secrets store):
echo "AURA_API_KEY=${AURA_API_KEY}" >> .env
echo "AGENT_API_TOKEN=${AGENT_API_TOKEN}" >> .env
docker compose -f docker-compose.prod.yml up -d
```

### Kubernetes Secret

```bash
kubectl create secret generic aura-secrets \
  --from-literal=AURA_API_KEY="$AURA_API_KEY" \
  --from-literal=AGENT_API_TOKEN="$AGENT_API_TOKEN" \
  -n aura
```

```yaml
# infra/k8s/deployment.yaml (excerpt)
envFrom:
  - secretRef:
      name: aura-secrets
```

### GitHub Actions — CI/CD Injection

Secrets are stored as [GitHub Actions repository secrets](https://docs.github.com/en/actions/security-for-github-actions/security-guides/using-secrets-in-github-actions) and injected at deploy time:

```yaml
# .github/workflows/deploy.yml (excerpt)
- name: Deploy to production
  env:
    AURA_API_KEY: ${{ secrets.AURA_API_KEY }}
    AGENT_API_TOKEN: ${{ secrets.AGENT_API_TOKEN }}
  run: |
    echo "AURA_API_KEY=${AURA_API_KEY}" >> .env
    docker compose -f docker-compose.prod.yml up -d
```

---

## Rotation Procedures

### AURA_API_KEY Rotation

`AURA_API_KEY` is used to authenticate calls to the AURA API server.

```bash
# 1. Generate a new key (32 random bytes, URL-safe base64):
NEW_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
echo "New key: $NEW_KEY"

# 2. Update the secret in your secrets store FIRST (before restarting the API):
#    GitHub Actions:
gh secret set AURA_API_KEY --body "$NEW_KEY"
#    Docker Swarm:
echo -n "$NEW_KEY" | docker secret create aura_api_key_v2 -

# 3. Update .env on all production hosts:
sed -i "s/^AURA_API_KEY=.*/AURA_API_KEY=${NEW_KEY}/" .env

# 4. Rolling restart of the API container (zero-downtime):
docker compose -f docker-compose.prod.yml up -d --no-deps api

# 5. Verify the new key works:
curl -H "Authorization: Bearer ${NEW_KEY}" http://localhost:8001/health

# 6. Notify all service consumers (n8n, agents) to update their stored token.

# 7. After confirming all consumers updated, invalidate the old key:
#    (For Docker Swarm) Remove the old secret version:
docker secret rm aura_api_key   # only after all services switched to v2
```

### AGENT_API_TOKEN Rotation

`AGENT_API_TOKEN` is used by external agents and n8n webhooks to submit goals.

```bash
# 1. Generate a new token:
NEW_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

# 2. Update GitHub Actions secret:
gh secret set AGENT_API_TOKEN --body "$NEW_TOKEN"

# 3. Update .env.n8n on production host:
sed -i "s/^AGENT_API_TOKEN=.*/AGENT_API_TOKEN=${NEW_TOKEN}/" .env.n8n

# 4. Update n8n workflow credential (via n8n UI or API):
#    Settings → Credentials → AURA API Token → update value

# 5. Restart dependent services:
docker compose -f docker-compose.prod.yml up -d --no-deps api

# 6. Re-test n8n → AURA webhook integration:
curl -X POST http://localhost:8001/api/v1/goal \
  -H "Authorization: Bearer ${NEW_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"goal": "test rotation", "dry_run": true}'
```

---

## Secret Scanning

| Tool | When | How |
|------|------|-----|
| `detect-secrets` pre-commit hook | Every commit | Blocks commit if pattern matched |
| GitHub Secret Scanning | Every push | Automatic; alerts via Security tab |
| `truffleHog` (CI step) | Every PR | `trufflehog git file://. --only-verified` |
| Manual audit | Quarterly | Review `.env.example`, Dockerfiles, CI configs |
