"""JWT authentication endpoints for AURA API.

Provides:
  POST /api/v1/auth/login   — issue access + refresh tokens
  POST /api/v1/auth/refresh — exchange refresh token for new access token
  POST /api/v1/auth/logout  — revoke a token (jti added to SQLite blocklist)

Security enforcements (NFR-S3):
  - Algorithm hard-coded to HS256; algorithm:none rejected at decode time
  - JWT_SECRET_KEY must be ≥32 bytes (enforced by AuthManager.__init__)
  - Access token lifetime capped at 24 h
  - Revocation persisted in SQLite (survives restarts)
"""

from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

_bearer = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    token: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_auth():
    """Return the global AuthManager, or raise 503 if not initialised."""
    try:
        from core.auth import get_auth_manager
        return get_auth_manager()
    except (RuntimeError, ImportError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Auth service unavailable: {exc}",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/login", response_model=TokenResponse, summary="Authenticate and obtain tokens")
async def login(body: LoginRequest) -> Dict[str, Any]:
    """Authenticate with username + password; returns access and refresh JWTs."""
    auth = _get_auth()
    user = auth.authenticate_user(body.username, body.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth.create_access_token(user)
    refresh_token = auth.create_refresh_token(user)
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=TokenResponse, summary="Refresh access token")
async def refresh(body: RefreshRequest) -> Dict[str, Any]:
    """Exchange a valid refresh token for a new access token.

    The old refresh token remains valid until it expires naturally.
    """
    auth = _get_auth()
    try:
        from core.auth import TokenError
        access_token = auth.refresh_access_token(body.refresh_token)
        new_refresh = auth.create_refresh_token(
            auth.get_current_user(access_token)
        )
        return TokenResponse(access_token=access_token, refresh_token=new_refresh)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token refresh failed: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, summary="Revoke a token")
async def logout(body: LogoutRequest) -> None:
    """Revoke the supplied token.  Its jti is added to the SQLite blocklist.

    Accepts either an access token or a refresh token.  After this call the
    token will be rejected by ``verify_token`` even before it expires.
    """
    auth = _get_auth()
    auth.revoke_token(body.token)
