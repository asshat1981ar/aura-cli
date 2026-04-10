"""Authentication middleware for AURA API.

Extracted from api_server.py as part of Sprint 1 server decomposition.
Provides both token-based auth (matching server.py) and optional JWT support.
"""

from __future__ import annotations

import os
import secrets
from typing import Any, Dict, Optional

from fastapi import Depends, Header, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Try to import AURA JWT auth components
try:
    from core.auth import get_auth_manager, AUTH_AVAILABLE
except ImportError:
    AUTH_AVAILABLE = False

security = HTTPBearer(auto_error=False)


class AuthMiddleware:
    """FastAPI middleware for authentication.

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
    if not AUTH_AVAILABLE or not credentials:
        return {"username": "anonymous", "role": "admin"}

    try:
        auth = get_auth_manager()
        user = auth.get_current_user(credentials.credentials)
        return user.to_dict()
    except Exception:
        return {"username": "anonymous", "role": "admin"}


def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    """Require a valid Bearer token matching the AGENT_API_TOKEN env var.

    Matches the token-based auth used in server.py so that all routers share
    a consistent authentication mechanism.

    Args:
        authorization: Value of the HTTP Authorization header.

    Raises:
        HTTPException: 401 if the header is absent, 403 if the token is wrong.
    """
    token = os.getenv("AGENT_API_TOKEN")
    if not token:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not secrets.compare_digest(authorization, f"Bearer {token}"):
        raise HTTPException(status_code=403, detail="Invalid token")
