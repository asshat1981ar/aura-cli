"""Tests for /health and /ready endpoints using FastAPI TestClient."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("AURA_SKIP_CHDIR", "1")
os.environ.pop("AGENT_API_TOKEN", None)

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client():
    """Return a TestClient with auth disabled."""
    os.environ.pop("AGENT_API_TOKEN", None)
    # Import fresh to avoid cached env state
    import importlib

    import aura_cli.server as server_mod

    importlib.reload(server_mod)
    with TestClient(server_mod.app, raise_server_exceptions=False) as c:
        yield c


# ── /health ──────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_status_healthy(self, client):
        data = resp = client.get("/health")
        assert resp.json()["status"] == "healthy"

    def test_health_returns_version(self, client):
        resp = client.get("/health")
        assert resp.json()["version"] == "0.1.0"

    def test_health_no_auth_required(self, client):
        """Health check must work without an Authorization header."""
        os.environ["AGENT_API_TOKEN"] = "super-secret"
        try:
            resp = client.get("/health")
            assert resp.status_code == 200
        finally:
            os.environ.pop("AGENT_API_TOKEN", None)

    def test_health_response_shape(self, client):
        data = client.get("/health").json()
        assert set(data.keys()) >= {"status", "version"}


# ── /ready ───────────────────────────────────────────────────────────────────


class TestReadyEndpoint:
    def test_ready_returns_200_when_db_readable(self, tmp_path, monkeypatch, client):
        """When SQLite DB is readable (we patch sqlite3.connect) returns 200."""
        import sqlite3

        mock_conn = MagicMock()
        mock_conn.execute.return_value = None

        monkeypatch.delenv("REDIS_URL", raising=False)

        import aura_cli.server as server_mod

        with patch.object(server_mod, "__name__", server_mod.__name__):
            # Patch sqlite3 connect inside the server module's namespace
            with patch("sqlite3.connect", return_value=mock_conn):
                resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_ready_returns_503_when_db_unavailable(self, monkeypatch, client):
        """When SQLite connect raises, /ready must return 503."""
        import sqlite3

        monkeypatch.delenv("REDIS_URL", raising=False)

        with patch("sqlite3.connect", side_effect=sqlite3.OperationalError("no such file")):
            resp = client.get("/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not_ready"
        assert "SQLite" in data["reason"]

    def test_ready_no_auth_required(self, monkeypatch, client):
        """Readiness probe must not require auth."""
        import sqlite3

        mock_conn = MagicMock()
        monkeypatch.delenv("REDIS_URL", raising=False)
        os.environ["AGENT_API_TOKEN"] = "secret"
        try:
            with patch("sqlite3.connect", return_value=mock_conn):
                resp = client.get("/ready")
            assert resp.status_code == 200
        finally:
            os.environ.pop("AGENT_API_TOKEN", None)

    def test_ready_503_when_redis_unavailable(self, monkeypatch, client):
        """When REDIS_URL is set but Redis is unreachable, /ready returns 503."""
        import sqlite3

        mock_conn = MagicMock()
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6399")

        mock_redis_client = MagicMock()
        mock_redis_client.ping.side_effect = Exception("Connection refused")

        with patch("sqlite3.connect", return_value=mock_conn):
            with patch("redis.from_url", return_value=mock_redis_client):
                resp = client.get("/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not_ready"
        assert "Redis" in data["reason"]

    def test_ready_200_when_redis_available(self, monkeypatch, client):
        """When REDIS_URL is set and Redis pings OK, /ready returns 200."""
        import sqlite3

        mock_conn = MagicMock()
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")

        mock_redis_client = MagicMock()
        mock_redis_client.ping.return_value = True

        with patch("sqlite3.connect", return_value=mock_conn):
            with patch("redis.from_url", return_value=mock_redis_client):
                resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_ready_skip_redis_when_no_env_var(self, monkeypatch, client):
        """When REDIS_URL is absent, Redis check is skipped entirely."""
        import sqlite3

        mock_conn = MagicMock()
        monkeypatch.delenv("REDIS_URL", raising=False)

        with patch("sqlite3.connect", return_value=mock_conn):
            with patch("redis.from_url") as mock_redis:
                resp = client.get("/ready")
                mock_redis.assert_not_called()
        assert resp.status_code == 200
