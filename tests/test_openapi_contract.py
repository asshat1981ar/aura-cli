"""OpenAPI contract test — keeps the FastAPI schema stable across changes.

On the first run (or when UPDATE_SNAPSHOTS=1 is set) the full schema is
written to ``tests/snapshots/openapi_schema.json``.  Subsequent runs load
that snapshot and compare against the live schema so that any breaking API
change is caught before merge.

Run with:
    python3 -m pytest tests/test_openapi_contract.py -v --no-cov

Regenerate snapshot:
    UPDATE_SNAPSHOTS=1 python3 -m pytest tests/test_openapi_contract.py -v --no-cov

Strict drift check (fails if live schema differs from snapshot):
    python3 -m pytest tests/test_openapi_contract.py -v --no-cov -k strict
    or set OPENAPI_STRICT=1 to enable strict mode for the snapshot test.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path / env bootstrap (mirrors the pattern used in test_server.py)
# ---------------------------------------------------------------------------
os.environ.setdefault("AURA_SKIP_CHDIR", "1")
os.environ.pop("AGENT_API_TOKEN", None)

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
_SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
_SNAPSHOT_FILE = _SNAPSHOT_DIR / "openapi_schema.json"

# Endpoints that must always appear in the published OpenAPI schema.
# NOTE: /metrics uses include_in_schema=False (Prometheus scrape target) so
#       it is intentionally absent from /openapi.json — use /execute (the
#       public RPC entry-point) and /discovery as the other anchor points.
_REQUIRED_PATHS = {"/health", "/tools", "/execute", "/discovery"}


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def openapi_client():
    """TestClient with auth disabled, module-scoped so we only boot once.

    Uses a clean import (no reload) to avoid prometheus_client double-
    registration errors when the module has already been imported by earlier
    test files in the same session.
    """
    os.environ.pop("AGENT_API_TOKEN", None)
    # If the module was already imported (e.g. by test_server.py running
    # first), reuse it — the app object is the same regardless.
    # If it hasn't been imported yet, import it fresh.
    import aura_cli.server as server_mod  # noqa: F401

    with TestClient(server_mod.app, raise_server_exceptions=False) as client:
        yield client


# ---------------------------------------------------------------------------
# Basic contract assertions
# ---------------------------------------------------------------------------
class TestOpenAPIEndpoint:
    """Validate that /openapi.json is reachable and structurally sound."""

    def test_openapi_json_returns_200(self, openapi_client):
        """GET /openapi.json must return HTTP 200."""
        resp = openapi_client.get("/openapi.json")
        assert resp.status_code == 200

    def test_openapi_json_is_valid_json(self, openapi_client):
        """Response body must be parseable as JSON."""
        resp = openapi_client.get("/openapi.json")
        # .json() raises ValueError on invalid JSON
        schema = resp.json()
        assert isinstance(schema, dict)

    def test_openapi_schema_has_required_top_level_keys(self, openapi_client):
        """Schema must contain the standard OpenAPI top-level fields."""
        schema = openapi_client.get("/openapi.json").json()
        for key in ("openapi", "info", "paths"):
            assert key in schema, f"Missing top-level key: {key!r}"

    def test_openapi_version_field_is_string(self, openapi_client):
        """The 'openapi' field must be a semver string (e.g. '3.1.0')."""
        schema = openapi_client.get("/openapi.json").json()
        assert isinstance(schema["openapi"], str)
        assert schema["openapi"].startswith("3.")

    def test_info_has_title_and_version(self, openapi_client):
        """The 'info' object must expose a title and version."""
        info = openapi_client.get("/openapi.json").json()["info"]
        assert "title" in info
        assert "version" in info


# ---------------------------------------------------------------------------
# Required-path assertions
# ---------------------------------------------------------------------------
class TestRequiredPaths:
    """Key endpoints must remain present in the published schema."""

    def test_required_paths_present(self, openapi_client):
        """All entries in _REQUIRED_PATHS must appear under 'paths'."""
        paths = openapi_client.get("/openapi.json").json().get("paths", {})
        missing = _REQUIRED_PATHS - set(paths.keys())
        assert not missing, f"Endpoints missing from OpenAPI schema: {missing}"

    @pytest.mark.parametrize("endpoint", sorted(_REQUIRED_PATHS))
    def test_each_required_path_individually(self, openapi_client, endpoint):
        """Parametrised so failures name the exact missing endpoint."""
        paths = openapi_client.get("/openapi.json").json().get("paths", {})
        assert endpoint in paths, f"Endpoint {endpoint!r} not found in OpenAPI paths"


# ---------------------------------------------------------------------------
# Snapshot regression test
# ---------------------------------------------------------------------------
class TestSnapshotRegression:
    """Detect unintentional changes to the published schema."""

    def _load_live_schema(self, client) -> str:
        """Return a canonical JSON string for the live schema."""
        schema = client.get("/openapi.json").json()
        return json.dumps(schema, sort_keys=True, indent=2)

    def test_snapshot(self, openapi_client):
        """Compare live schema against the committed snapshot.

        - First run / UPDATE_SNAPSHOTS=1 → write snapshot, pass.
        - Subsequent runs                → compare; fail on diff.
        """
        live_str = self._load_live_schema(openapi_client)
        update = os.getenv("UPDATE_SNAPSHOTS", "").strip() == "1"

        if not _SNAPSHOT_FILE.exists() or update:
            _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            _SNAPSHOT_FILE.write_text(live_str, encoding="utf-8")
            # Either first run or explicit refresh — always passes.
            return

        saved_str = _SNAPSHOT_FILE.read_text(encoding="utf-8")
        if live_str != saved_str:
            # Produce a readable diff in the failure message.
            live_lines = live_str.splitlines(keepends=True)
            saved_lines = saved_str.splitlines(keepends=True)
            import difflib

            diff = "".join(
                difflib.unified_diff(
                    saved_lines,
                    live_lines,
                    fromfile="snapshot (expected)",
                    tofile="live /openapi.json",
                    n=5,
                )
            )
            strict = os.getenv("OPENAPI_STRICT", "").strip() == "1"
            msg = (
                "OpenAPI schema has drifted from snapshot "
                "(--strict mode: OPENAPI_STRICT=1).\n\n"
                if strict
                else "OpenAPI schema has changed — run with UPDATE_SNAPSHOTS=1 to accept.\n\n"
            )
            pytest.fail(msg + diff)


# ---------------------------------------------------------------------------
# New tests: all server routes appear in schema
# ---------------------------------------------------------------------------


class TestAllServerEndpointsInSchema:
    """Every route registered on the FastAPI app must appear in the published schema.

    This catches endpoints added to ``aura_cli/server.py`` that were not
    reflected in the OpenAPI schema (e.g. missing ``include_in_schema``
    handling or accidental exclusion).

    Endpoints explicitly excluded from the schema (``include_in_schema=False``)
    are allowed to be absent — we collect those and skip them here.
    """

    def test_all_server_endpoints_in_schema(self, openapi_client):
        """Every include_in_schema route in app.routes must appear in /openapi.json."""
        import aura_cli.server as server_mod
        from starlette.routing import Route

        schema_paths = set(
            openapi_client.get("/openapi.json").json().get("paths", {}).keys()
        )

        missing = []
        for route in server_mod.app.routes:
            if not isinstance(route, Route):
                continue
            # Routes flagged include_in_schema=False are intentionally absent
            if not getattr(route, "include_in_schema", True):
                continue
            # Starlette internal routes (e.g. /openapi.json, /docs, /redoc)
            if route.path.startswith("/openapi") or route.path in ("/docs", "/redoc"):
                continue
            if route.path not in schema_paths:
                missing.append(route.path)

        assert not missing, (
            f"Routes registered in app.routes but absent from OpenAPI schema: {missing}\n"
            "If a route is intentionally excluded set include_in_schema=False on it."
        )


# ---------------------------------------------------------------------------
# New tests: response schema validation for key endpoints
# ---------------------------------------------------------------------------


class TestHealthResponseSchema:
    """GET /health must return a JSON object matching the schema definition."""

    def test_health_response_schema(self, openapi_client):
        """GET /health → 200 with JSON object containing at least 'status'."""
        resp = openapi_client.get("/health")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        body = resp.json()
        assert isinstance(body, dict), "Response body must be a JSON object"
        assert "status" in body, (
            f"'status' key missing from /health response; got keys: {list(body.keys())}"
        )

    def test_health_response_schema_matches_openapi_definition(self, openapi_client):
        """Response schema for /health must be declared as additionalProperties:true object."""
        schema = openapi_client.get("/openapi.json").json()
        health_get = schema["paths"]["/health"]["get"]
        response_schema = (
            health_get["responses"]["200"]["content"]["application/json"]["schema"]
        )
        # FastAPI declares open-ended dict endpoints as {type: object, additionalProperties: true}
        assert response_schema.get("type") == "object", (
            f"/health 200 response schema should have type 'object', got: {response_schema}"
        )


class TestExecuteRequestSchema:
    """POST /execute must accept the schema-documented request body."""

    def test_execute_accepts_documented_request_body(self, openapi_client):
        """POST /execute with a valid ExecuteRequest body must not return 422."""
        # Load the documented schema to confirm the shape we're testing against
        schema = openapi_client.get("/openapi.json").json()
        execute_ref = (
            schema["paths"]["/execute"]["post"]["requestBody"]["content"]
            ["application/json"]["schema"]
        )
        # Must reference the ExecuteRequest component (or be inline with tool_name)
        assert "$ref" in execute_ref or "properties" in execute_ref, (
            f"/execute requestBody schema missing '$ref' or 'properties': {execute_ref}"
        )

        # Send the minimal documented request body: {"tool_name": "env"}
        resp = openapi_client.post(
            "/execute",
            json={"tool_name": "env", "args": []},
        )
        # Any status other than 422 (Unprocessable Entity) proves the schema is accepted.
        # 200, 500, etc. are all acceptable — we only care that FastAPI parsed the body.
        assert resp.status_code != 422, (
            f"POST /execute with valid body returned 422; body: {resp.text}"
        )

    def test_execute_rejects_missing_tool_name(self, openapi_client):
        """POST /execute without required 'tool_name' must return 422."""
        resp = openapi_client.post("/execute", json={"args": []})
        assert resp.status_code == 422, (
            f"Expected 422 for missing tool_name, got {resp.status_code}"
        )
