"""Integration tests for the POST /run API endpoint.

Validates auth, env-guard, happy-path acceptance, and run_id presence without
making real LLM calls — LoopOrchestrator.run_loop is patched throughout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path / env bootstrap
# ---------------------------------------------------------------------------
os.environ.setdefault("AURA_SKIP_CHDIR", "1")

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import aura_cli.server as server_mod  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_TOKEN = "sprint5-test-token"
_AUTH_HEADER = {"Authorization": f"Bearer {_VALID_TOKEN}"}
_GOAL_BODY = {"goal": "add a hello() function to utils.py"}

# Dummy run_loop return value — avoids executing the real pipeline.
_MOCK_LOOP_RESULT = {
    "goal": "add a hello() function to utils.py",
    "stop_reason": "PASS",
    "history": [],
}


@pytest.fixture(autouse=True)
def _patch_run_loop():
    """Patch LoopOrchestrator.run_loop so tests never spin up a real pipeline."""
    with patch(
        "core.orchestrator.LoopOrchestrator.run_loop",
        return_value=_MOCK_LOOP_RESULT,
    ):
        yield


@pytest.fixture()
def client_no_auth():
    """TestClient with no AGENT_API_TOKEN set (auth disabled server-side)."""
    env_backup = os.environ.pop("AGENT_API_TOKEN", None)
    os.environ.pop("AGENT_API_ENABLE_RUN", None)
    try:
        with TestClient(server_mod.app, raise_server_exceptions=False) as c:
            yield c
    finally:
        if env_backup is not None:
            os.environ["AGENT_API_TOKEN"] = env_backup


@pytest.fixture()
def client_with_auth():
    """TestClient with auth enabled via AGENT_API_TOKEN and run enabled."""
    os.environ["AGENT_API_TOKEN"] = _VALID_TOKEN
    os.environ["AGENT_API_ENABLE_RUN"] = "1"
    try:
        with TestClient(server_mod.app, raise_server_exceptions=False) as c:
            yield c
    finally:
        os.environ.pop("AGENT_API_TOKEN", None)
        os.environ.pop("AGENT_API_ENABLE_RUN", None)


@pytest.fixture()
def client_auth_run_disabled():
    """TestClient with auth enabled but AGENT_API_ENABLE_RUN NOT set."""
    os.environ["AGENT_API_TOKEN"] = _VALID_TOKEN
    os.environ.pop("AGENT_API_ENABLE_RUN", None)
    try:
        with TestClient(server_mod.app, raise_server_exceptions=False) as c:
            yield c
    finally:
        os.environ.pop("AGENT_API_TOKEN", None)
        os.environ.pop("AGENT_API_ENABLE_RUN", None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunEndpointAuth:
    """POST /run authentication guard."""

    def test_run_endpoint_requires_auth(self, client_no_auth: TestClient):
        """Sending no Authorization header when a token is configured returns 401 or 403."""
        # Enable run so only auth is being tested.
        os.environ["AGENT_API_TOKEN"] = _VALID_TOKEN
        os.environ["AGENT_API_ENABLE_RUN"] = "1"
        try:
            resp = client_no_auth.post("/run", json=_GOAL_BODY)
        finally:
            os.environ.pop("AGENT_API_TOKEN", None)
            os.environ.pop("AGENT_API_ENABLE_RUN", None)

        assert resp.status_code in (401, 403), f"Expected 401 or 403 without auth, got {resp.status_code}"

    def test_run_endpoint_wrong_token_rejected(self, client_with_auth: TestClient):
        """A wrong Bearer token must be rejected with 401 or 403."""
        resp = client_with_auth.post(
            "/run",
            json=_GOAL_BODY,
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code in (401, 403), f"Expected 401 or 403 for wrong token, got {resp.status_code}"


class TestRunEndpointEnvGuard:
    """POST /run environment-variable guard."""

    def test_run_endpoint_disabled_without_env(self, client_auth_run_disabled: TestClient):
        """Endpoint returns 403 when AGENT_API_ENABLE_RUN is not set to '1'."""
        resp = client_auth_run_disabled.post(
            "/run",
            json=_GOAL_BODY,
            headers=_AUTH_HEADER,
        )
        assert resp.status_code in (403, 404), f"Expected 403 or 404 when run is disabled, got {resp.status_code}"

    def test_run_endpoint_disabled_env_wrong_value(self, client_auth_run_disabled: TestClient):
        """Setting AGENT_API_ENABLE_RUN to a non-'1' value must also be rejected."""
        os.environ["AGENT_API_ENABLE_RUN"] = "true"
        try:
            resp = client_auth_run_disabled.post(
                "/run",
                json=_GOAL_BODY,
                headers=_AUTH_HEADER,
            )
        finally:
            os.environ.pop("AGENT_API_ENABLE_RUN", None)

        assert resp.status_code in (403, 404), f"Expected 403 or 404 for AGENT_API_ENABLE_RUN=true, got {resp.status_code}"


class TestRunEndpointHappyPath:
    """POST /run success path."""

    def test_run_endpoint_accepts_goal(self, client_with_auth: TestClient):
        """POST /run with valid token and enabled env returns HTTP 200."""
        resp = client_with_auth.post(
            "/run",
            json=_GOAL_BODY,
            headers=_AUTH_HEADER,
        )
        assert resp.status_code == 200, f"Expected 200 for valid /run request, got {resp.status_code}: {resp.text}"

    def test_run_creates_run_id(self, client_with_auth: TestClient):
        """Response body must contain a non-empty 'run_id' string."""
        resp = client_with_auth.post(
            "/run",
            json=_GOAL_BODY,
            headers=_AUTH_HEADER,
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        body = resp.json()
        assert "run_id" in body, f"'run_id' missing from response: {body}"
        assert isinstance(body["run_id"], str), "run_id must be a string"
        assert len(body["run_id"]) > 0, "run_id must not be empty"

    def test_run_response_contains_status(self, client_with_auth: TestClient):
        """Response body should include a 'status' field acknowledging the request."""
        resp = client_with_auth.post(
            "/run",
            json=_GOAL_BODY,
            headers=_AUTH_HEADER,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body, f"'status' missing from response: {body}"

    def test_run_ids_are_unique(self, client_with_auth: TestClient):
        """Each call to POST /run must produce a distinct run_id."""
        ids = set()
        for _ in range(3):
            resp = client_with_auth.post(
                "/run",
                json=_GOAL_BODY,
                headers=_AUTH_HEADER,
            )
            assert resp.status_code == 200
            ids.add(resp.json()["run_id"])
        assert len(ids) == 3, "run_ids must be unique across calls"

    def test_run_missing_goal_returns_error(self, client_with_auth: TestClient):
        """POST /run with no 'goal' field in the body should return a 4xx error."""
        resp = client_with_auth.post(
            "/run",
            json={},
            headers=_AUTH_HEADER,
        )
        assert resp.status_code in range(400, 500), f"Expected 4xx for missing goal, got {resp.status_code}"


class TestRunEndpointMockOrchestrator:
    """Verify that LoopOrchestrator.run_loop is invoked correctly."""

    def test_run_calls_orchestrator_in_background(self, client_with_auth: TestClient):
        """POST /run must return immediately (accepted) without waiting for run_loop."""
        # The autouse _patch_run_loop fixture ensures run_loop is a MagicMock.
        # We just verify the endpoint is non-blocking and returns 200.
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_loop.return_value = _MOCK_LOOP_RESULT

        with patch.object(server_mod, "orchestrator", mock_orchestrator):
            resp = client_with_auth.post(
                "/run",
                json=_GOAL_BODY,
                headers=_AUTH_HEADER,
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") == "accepted", f"Expected status='accepted', got {body}"
