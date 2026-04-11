# AURA CLI — Release Checklist

This document defines the steps required to cut a production release of AURA CLI.
Follow each section in order. Check off items as you complete them.

---

## Pre-Release: Code Freeze

- [ ] All feature PRs for this release merged to `main`
- [ ] No open `BLOCKER` or `P0` issues in the milestone
- [ ] `CHANGELOG.md` updated with release version and date (replace `[Unreleased]` with `[X.Y.Z] - YYYY-MM-DD`)
- [ ] Version bumped in `pyproject.toml` (`version = "X.Y.Z"`)
- [ ] `docs/adr/INDEX.md` updated with any new ADRs from this release cycle

---

## Quality Gates

Run all of these locally before tagging:

```bash
# 1. Lint — must be clean
python3 -m ruff check . && python3 -m ruff format . --check

# 2. SAST — no HIGH findings
bandit -r aura_cli core agents memory -c pyproject.toml --severity-level high -q

# 3. Dependency audit — no known CVEs
pip-audit --requirement requirements.txt --vulnerability-service pypi

# 4. Full safe test suite — must be 0 failures
python3 -m pytest \
  tests/test_auth.py tests/test_jwt_hardening.py tests/test_server_api.py \
  tests/test_sanitizer.py tests/test_correlation.py tests/test_config_schema.py \
  tests/test_sandbox_unit.py tests/test_rate_limit.py tests/test_db_migrations.py \
  tests/test_agents_planner.py tests/test_agents_critic.py tests/test_agents_verifier.py \
  tests/test_agents_applicator.py tests/test_orchestrator.py tests/test_model_adapter.py \
  tests/test_memory_brain.py tests/test_memory_store.py tests/test_memory_controller.py \
  tests/test_handlers.py \
  --cov=aura_cli --cov=core --cov=agents --cov=memory \
  --timeout=30 -q

# 5. Config validation
python3 scripts/validate_config.py

# 6. OpenAPI spec current
python3 scripts/export_openapi.py
git diff --exit-code docs/api/openapi.json
```

- [ ] Ruff lint: **0 errors**
- [ ] Bandit SAST: **0 HIGH severity findings**
- [ ] pip-audit: **0 unpatched CVEs** (or documented exceptions in `docs/security/`)
- [ ] Test suite: **all tests passing**, coverage ≥ `fail_under` in `pyproject.toml`
- [ ] Config validation: **no errors**
- [ ] OpenAPI spec: **no drift** from committed `docs/api/openapi.json`

---

## Docker Build Verification

```bash
# Build image (must succeed with 0 errors)
docker build -t aura-cli:release-candidate .

# Smoke test: server starts and responds
docker run --rm -d -p 8000:8000 --name aura-test aura-cli:release-candidate
sleep 5
curl -sf http://localhost:8000/health || (docker logs aura-test && exit 1)
curl -sf http://localhost:8000/ready || true   # may fail without full config
docker stop aura-test
```

- [ ] Docker image builds successfully (no errors)
- [ ] Health endpoint responds `{"status": "ok"}` in container
- [ ] Image size reasonable (check with `docker images aura-cli:release-candidate`)

---

## Tagging & Release

```bash
# Ensure on main and up to date
git checkout main && git pull

# Create annotated tag
git tag -a vX.Y.Z -m "Release vX.Y.Z — <one-line summary>"
git push origin vX.Y.Z
```

- [ ] Annotated tag `vX.Y.Z` pushed to origin
- [ ] GitHub release created from tag (via UI or `gh release create vX.Y.Z --notes-from-tag`)
- [ ] Release workflow (`.github/workflows/release.yml`) triggered and completed

---

## Post-Release Verification

- [ ] CI pipeline passes on the tagged commit
- [ ] PyPI package published (if applicable) — version matches `pyproject.toml`
- [ ] Docker image pushed to registry (if applicable)
- [ ] `main` branch HEAD is the released commit (no post-tag commits on main)
- [ ] GitHub release notes include CHANGELOG section for this version

---

## Post-Release Housekeeping

- [ ] Bump version in `pyproject.toml` to next dev version (e.g., `1.0.1.dev0`)
- [ ] Add `## [Unreleased]` section back to `CHANGELOG.md`
- [ ] Create milestone for next release in GitHub Issues
- [ ] Close current milestone (all issues resolved or deferred)

---

## Rollback Procedure

If a critical defect is found after release:

1. **Hotfix branch**: `git checkout -b hotfix/X.Y.Z-fix-description vX.Y.Z`
2. Apply fix, run all quality gates (above)
3. Bump to `X.Y.Z+1` (patch increment)
4. Merge hotfix branch to `main` via PR
5. Tag `vX.Y.Z+1` and follow tagging steps above
6. Update GitHub release notes to flag the issue and point to the hotfix

---

## Known Issues / Exceptions

Document any quality gate exceptions here (CVEs with no fix available, accepted risk items):

| Issue | Severity | Justification | Expires |
|-------|----------|---------------|---------|
| _(none)_ | — | — | — |
