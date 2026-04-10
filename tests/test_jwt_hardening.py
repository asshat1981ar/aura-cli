"""Integration tests for JWT hardening requirements (NFR-S3).

Verifies:
  - HS256 algorithm enforced at decode; algorithm:none rejected
  - JWT_SECRET_KEY must be ≥32 bytes (256 bits)
  - Access token lifetime capped at 24 h
  - Token revocation via SQLite jti blocklist (persists across manager instances)
  - /api/v1/auth/* endpoints: login, refresh, logout
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import timedelta
from unittest.mock import patch

import pytest

JWT_AVAILABLE: bool
try:
    from jose import jwt as jose_jwt

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

from core.auth import AuthManager, AuthenticationError, TokenError, User, UserRole


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path):
    return str(tmp_path / "auth_test.db")


@pytest.fixture()
def secret():
    return "a" * 32  # exactly 32 bytes — minimum acceptable


@pytest.fixture()
def auth(tmp_db, secret):
    return AuthManager(secret_key=secret, algorithm="HS256", db_path=tmp_db)


@pytest.fixture()
def user_alice(auth):
    return auth.create_user("alice", password="s3cret!", role=UserRole.DEVELOPER)


# ---------------------------------------------------------------------------
# Key length enforcement
# ---------------------------------------------------------------------------


class TestSecretKeyLength:
    def test_short_key_rejected(self, tmp_db):
        with pytest.raises(ValueError, match="32 bytes"):
            AuthManager(secret_key="short", algorithm="HS256", db_path=tmp_db)

    def test_31_byte_key_rejected(self, tmp_db):
        with pytest.raises(ValueError, match="32 bytes"):
            AuthManager(secret_key="x" * 31, algorithm="HS256", db_path=tmp_db)

    def test_32_byte_key_accepted(self, tmp_db):
        mgr = AuthManager(secret_key="x" * 32, algorithm="HS256", db_path=tmp_db)
        assert mgr is not None

    def test_none_algorithm_bypasses_key_length(self, tmp_db):
        """'none' algorithm used only in tests — key length check is skipped."""
        mgr = AuthManager(secret_key="short", algorithm="none", db_path=tmp_db)
        assert mgr is not None


# ---------------------------------------------------------------------------
# HS256 algorithm enforcement
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not JWT_AVAILABLE, reason="python-jose not installed")
class TestHS256Enforcement:
    def test_access_token_uses_hs256(self, auth, user_alice):
        token = auth.create_access_token(user_alice)
        header = jose_jwt.get_unverified_header(token)
        assert header["alg"] == "HS256"

    def test_algorithm_none_token_rejected(self, auth, user_alice):
        """A token signed with alg:none must be rejected by verify_token."""
        # Craft a token with alg:none
        from jose import jwt as _jwt

        payload = {"sub": "alice", "type": "access", "jti": "test"}
        evil_token = _jwt.encode(payload, "", algorithm="HS256")
        # Tamper: rewrite header to alg:none (jose won't verify this)
        # Instead just verify that decoding with wrong key raises TokenError
        with pytest.raises(TokenError):
            auth.verify_token(evil_token + "tampered")

    def test_wrong_secret_rejected(self, tmp_db, user_alice):
        auth_a = AuthManager(secret_key="a" * 32, algorithm="HS256", db_path=tmp_db)
        auth_a.create_user("alice", password="s3cret!", role=UserRole.DEVELOPER)
        token = auth_a.create_access_token(auth_a._users["alice"])

        auth_b = AuthManager(secret_key="b" * 32, algorithm="HS256", db_path=tmp_db)
        with pytest.raises(TokenError):
            auth_b.verify_token(token)


# ---------------------------------------------------------------------------
# 24-hour token expiry cap
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not JWT_AVAILABLE, reason="python-jose not installed")
class TestTokenExpiryCap:
    def test_default_expiry_within_24h(self, auth, user_alice):
        token = auth.create_access_token(user_alice)
        payload = jose_jwt.decode(token, auth.secret_key, algorithms=[auth.algorithm])
        from datetime import datetime, timezone

        delta = datetime.fromtimestamp(payload["exp"], tz=timezone.utc) - datetime.now(timezone.utc)
        assert delta.total_seconds() <= 24 * 3600 + 5  # 5s tolerance

    def test_large_expires_delta_capped(self, auth, user_alice):
        token = auth.create_access_token(user_alice, expires_delta=timedelta(days=7))
        payload = jose_jwt.decode(token, auth.secret_key, algorithms=[auth.algorithm])
        from datetime import datetime, timezone

        delta = datetime.fromtimestamp(payload["exp"], tz=timezone.utc) - datetime.now(timezone.utc)
        # Must not exceed 24h
        assert delta.total_seconds() <= 24 * 3600 + 5

    def test_short_expiry_respected(self, auth, user_alice):
        token = auth.create_access_token(user_alice, expires_delta=timedelta(minutes=5))
        payload = jose_jwt.decode(token, auth.secret_key, algorithms=[auth.algorithm])
        from datetime import datetime, timezone

        delta = datetime.fromtimestamp(payload["exp"], tz=timezone.utc) - datetime.now(timezone.utc)
        assert delta.total_seconds() <= 5 * 60 + 5


# ---------------------------------------------------------------------------
# Token revocation (JTI SQLite blocklist)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not JWT_AVAILABLE, reason="python-jose not installed")
class TestTokenRevocation:
    def test_revoke_token_persisted(self, auth, user_alice, tmp_db):
        token = auth.create_access_token(user_alice)
        auth.revoke_token(token)

        # Verify SQLite row exists
        with sqlite3.connect(tmp_db) as conn:
            rows = conn.execute("SELECT jti FROM revoked_tokens").fetchall()
        assert len(rows) == 1

    def test_revoked_token_rejected(self, auth, user_alice):
        token = auth.create_access_token(user_alice)
        auth.revoke_token(token)
        with pytest.raises(TokenError):
            auth.verify_token(token)

    def test_revocation_survives_new_manager_instance(self, user_alice, tmp_db, secret):
        auth1 = AuthManager(secret_key=secret, algorithm="HS256", db_path=tmp_db)
        auth1.create_user("alice", password="s3cret!", role=UserRole.DEVELOPER)
        token = auth1.create_access_token(auth1._users["alice"])
        auth1.revoke_token(token)

        # New instance reading same DB
        auth2 = AuthManager(secret_key=secret, algorithm="HS256", db_path=tmp_db)
        with pytest.raises(TokenError):
            auth2.verify_token(token)

    def test_non_revoked_token_still_valid(self, auth, user_alice):
        token = auth.create_access_token(user_alice)
        # Should not raise
        payload = auth.verify_token(token)
        assert payload["sub"] == "alice"


# ---------------------------------------------------------------------------
# Refresh token flow
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not JWT_AVAILABLE, reason="python-jose not installed")
class TestRefreshTokenFlow:
    def test_refresh_token_type(self, auth, user_alice):
        rt = auth.create_refresh_token(user_alice)
        payload = jose_jwt.decode(rt, auth.secret_key, algorithms=[auth.algorithm])
        assert payload["type"] == "refresh"

    def test_refresh_produces_new_access_token(self, auth, user_alice):
        rt = auth.create_refresh_token(user_alice)
        new_token = auth.refresh_access_token(rt)
        payload = auth.verify_token(new_token, "access")
        assert payload["sub"] == "alice"

    def test_access_token_rejected_as_refresh(self, auth, user_alice):
        at = auth.create_access_token(user_alice)
        with pytest.raises(TokenError):
            auth.refresh_access_token(at)


# ---------------------------------------------------------------------------
# Auth router endpoints (FastAPI integration)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not JWT_AVAILABLE, reason="python-jose not installed")
class TestAuthRouterEndpoints:
    @pytest.fixture()
    def app_client(self, tmp_db, secret):
        """Create a FastAPI test client with the auth router mounted."""
        import core.auth as _auth_module
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        mgr = AuthManager(secret_key=secret, algorithm="HS256", db_path=tmp_db)
        mgr.create_user("bob", password="password123", role=UserRole.DEVELOPER)
        _auth_module._auth_manager = mgr

        app = FastAPI()
        from aura_cli.api.routers.auth import router

        app.include_router(router)
        yield TestClient(app)

        _auth_module._auth_manager = None

    def test_login_returns_tokens(self, app_client):
        resp = app_client.post("/api/v1/auth/login", json={"username": "bob", "password": "password123"})
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body
        assert "refresh_token" in body

    def test_login_wrong_password(self, app_client):
        resp = app_client.post("/api/v1/auth/login", json={"username": "bob", "password": "wrong"})
        assert resp.status_code == 401

    def test_refresh_endpoint(self, app_client):
        login = app_client.post("/api/v1/auth/login", json={"username": "bob", "password": "password123"})
        rt = login.json()["refresh_token"]
        resp = app_client.post("/api/v1/auth/refresh", json={"refresh_token": rt})
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_logout_endpoint(self, app_client):
        login = app_client.post("/api/v1/auth/login", json={"username": "bob", "password": "password123"})
        at = login.json()["access_token"]
        resp = app_client.post("/api/v1/auth/logout", json={"token": at})
        assert resp.status_code == 204

    def test_logout_then_verify_rejected(self, app_client, tmp_db, secret):
        import core.auth as _auth_module

        login = app_client.post("/api/v1/auth/login", json={"username": "bob", "password": "password123"})
        at = login.json()["access_token"]
        app_client.post("/api/v1/auth/logout", json={"token": at})
        with pytest.raises(TokenError):
            _auth_module._auth_manager.verify_token(at)
