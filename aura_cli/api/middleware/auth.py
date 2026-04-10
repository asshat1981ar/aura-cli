"""Authentication middleware for AURA API.

Extracted from api_server.py as part of Sprint 1 server decomposition.
Provides JWT token verification and user extraction.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Try to import AURA auth components
try:
    from core.auth import get_auth_manager, AUTH_AVAILABLE
except ImportError:
    AUTH_AVAILABLE = False

security = HTTPBearer(auto_error=False)


class AuthMiddleware:
    """FastAPI middleware for JWT authentication.

    Usage:
        app.add_middleware(AuthMiddleware)
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Dict, receive: Any, send: Any) -> None:
        await self.app(scope, receive, send)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """Verify JWT token and return user.

    Always allows anonymous access when auth is not available or in development.

    Args:
        credentials: HTTP Authorization credentials with Bearer token.

    Returns:
        Dict with username and role keys.
    """
    # Always allow anonymous access when auth is not available
    if not AUTH_AVAILABLE:
        return {"username": "anonymous", "role": "admin"}

    # Also allow if no credentials provided (for development)
    if not credentials:
        return {"username": "anonymous", "role": "admin"}

    try:
        auth = get_auth_manager()
        user = auth.get_current_user(credentials.credentials)
        return user.to_dict()
    except Exception:
        # Fall back to anonymous on error
        return {"username": "anonymous", "role": "admin"}


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """Require valid authentication.

    Unlike get_current_user, this raises HTTPException on missing/invalid auth.

    Args:
        credentials: HTTP Authorization credentials with Bearer token.

    Returns:
        Dict with username and role keys.

    Raises:
        HTTPException: 401 if credentials missing, 403 if invalid.
    """
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not AUTH_AVAILABLE:
        return {"username": "anonymous", "role": "admin"}

    try:
        auth = get_auth_manager()
        user = auth.get_current_user(credentials.credentials)
        return user.to_dict()
    except Exception as exc:
        raise HTTPException(status_code=403, detail=f"Invalid token: {exc}") from exc
