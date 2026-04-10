"""MCP Server API Key Authentication Module.

Provides secure API key validation for all MCP servers with:
- Constant-time comparison using hmac.compare_digest (timing attack resistant)
- X-API-Key header support
- Config-based key loading from ConfigManager
- Backward compatibility (auth optional if key not configured)

Usage:
    from tools.mcp_auth import APIKeyMiddleware, get_api_key_validator

    # Add middleware to FastAPI app
    app.add_middleware(APIKeyMiddleware, server_name="dev_tools")

    # Or use as a dependency
    async def endpoint(auth: str = Depends(get_api_key_validator("dev_tools"))):
        ...
"""

from __future__ import annotations

import hmac
import os
from typing import Optional

from fastapi import Header, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# ---------------------------------------------------------------------------
# Key loading and validation
# ---------------------------------------------------------------------------

def get_mcp_server_api_key(server_name: str) -> Optional[str]:
    """Load API key for a named MCP server from configuration.
    
    Priority:
    1. Environment variable: MCP_<SERVER_NAME>_API_KEY (uppercase)
    2. Config manager: mcp_server_api_keys.<server_name>
    3. Legacy fallback: MCP_API_TOKEN (for dev_tools only)
    
    Args:
        server_name: One of "dev_tools", "skills", "control", 
                     "agentic_loop", "copilot"
    
    Returns:
        API key string or None if not configured
    """
    # 1. Check environment variable first
    env_var = f"MCP_{server_name.upper()}_API_KEY"
    env_key = os.getenv(env_var, "").strip()
    if env_key:
        return env_key
    
    # 2. Check config manager
    try:
        from core.config_manager import config as _cfg
        cfg_key = _cfg.get_mcp_server_api_key(server_name)
        if cfg_key:
            return cfg_key
    except Exception:
        pass
    
    # 3. Legacy fallback for dev_tools
    if server_name == "dev_tools":
        legacy_token = os.getenv("MCP_API_TOKEN", "").strip()
        if legacy_token:
            return legacy_token
    
    return None


def validate_api_key(server_name: str, provided_key: Optional[str]) -> bool:
    """Validate an API key using constant-time comparison.
    
    Args:
        server_name: Name of the MCP server
        provided_key: The API key provided in the request
    
    Returns:
        True if valid, False otherwise
    """
    expected_key = get_mcp_server_api_key(server_name)
    
    # If no key is configured, authentication is optional (backward compatible)
    if not expected_key:
        return True
    
    # If a key is configured, the request must provide one
    if not provided_key:
        return False
    
    # Constant-time comparison to prevent timing attacks
    return hmac.compare_digest(provided_key, expected_key)


# ---------------------------------------------------------------------------
# FastAPI Dependency
# ---------------------------------------------------------------------------

def get_api_key_validator(server_name: str):
    """Create a FastAPI dependency for API key validation.
    
    Usage:
        from fastapi import Depends
        from tools.mcp_auth import get_api_key_validator
        
        @app.get("/endpoint")
        async def endpoint(auth: str = Depends(get_api_key_validator("dev_tools"))):
            return {"message": "Authenticated"}
    
    Args:
        server_name: Name of the MCP server
    
    Returns:
        A dependency function that validates X-API-Key header
    """
    def _validate(
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
        authorization: Optional[str] = Header(default=None),
    ) -> str:
        """Validate API key from X-API-Key header or Authorization header.
        
        Supports:
        - X-API-Key: <key>
        - Authorization: Bearer <key>
        """
        provided_key = None
        
        # Check X-API-Key header first (preferred)
        if x_api_key:
            provided_key = x_api_key
        # Fall back to Authorization: Bearer <key>
        elif authorization and authorization.lower().startswith("bearer "):
            provided_key = authorization[7:].strip()
        
        # Check if authentication is required
        expected_key = get_mcp_server_api_key(server_name)
        if not expected_key:
            # Auth not configured, allow through
            return "optional"
        
        # Validate the provided key
        if not provided_key:
            raise HTTPException(
                status_code=401,
                detail="Missing API key. Provide via X-API-Key header or Authorization: Bearer <key>",
            )
        
        if not hmac.compare_digest(provided_key, expected_key):
            raise HTTPException(
                status_code=403,
                detail="Invalid API key",
            )
        
        return provided_key
    
    return _validate


# ---------------------------------------------------------------------------
# FastAPI Middleware
# ---------------------------------------------------------------------------

class APIKeyMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for API key authentication.
    
    Usage:
        from fastapi import FastAPI
        from tools.mcp_auth import APIKeyMiddleware
        
        app = FastAPI()
        app.add_middleware(APIKeyMiddleware, server_name="dev_tools")
    
    The middleware will:
    - Skip auth for health check endpoints (configurable)
    - Require X-API-Key or Authorization header if key is configured
    - Allow requests through if no key is configured (backward compatible)
    """
    
    def __init__(
        self,
        app,
        server_name: str,
        exempt_paths: Optional[set] = None,
        header_name: str = "X-API-Key",
    ):
        """Initialize the middleware.
        
        Args:
            app: FastAPI application
            server_name: Name of the MCP server
            exempt_paths: Set of path prefixes that don't require auth (default: {"/health", "/"})
            header_name: Name of the API key header
        """
        super().__init__(app)
        self.server_name = server_name
        self.exempt_paths = exempt_paths or {"/health", "/"}
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next):
        """Process the request and validate API key if required."""
        path = request.url.path
        
        # Check if path is exempt
        if any(path == exempt or path.startswith(exempt + "/") 
               for exempt in self.exempt_paths):
            return await call_next(request)
        
        # Get configured key
        expected_key = get_mcp_server_api_key(self.server_name)
        
        # If no key configured, allow through (backward compatible)
        if not expected_key:
            return await call_next(request)
        
        # Get provided key from headers
        provided_key = None
        
        # Check X-API-Key header
        x_api_key = request.headers.get(self.header_name)
        if x_api_key:
            provided_key = x_api_key
        else:
            # Check Authorization: Bearer <key>
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.lower().startswith("bearer "):
                provided_key = auth_header[7:].strip()
        
        # Validate
        if not provided_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Missing API key",
                    "detail": f"Provide via {self.header_name} header or Authorization: Bearer <key>",
                },
            )
        
        if not hmac.compare_digest(provided_key, expected_key):
            return JSONResponse(
                status_code=403,
                content={"error": "Invalid API key"},
            )
        
        # Key is valid, proceed
        return await call_next(request)


# ---------------------------------------------------------------------------
# Legacy compatibility: Update existing auth functions
# ---------------------------------------------------------------------------

def patch_server_auth(server_name: str, existing_token: Optional[str] = None) -> Optional[str]:
    """Get the effective API key for a server, considering legacy tokens.
    
    This helper is used when migrating existing servers to use the new
    centralized auth system.
    
    Args:
        server_name: Name of the MCP server
        existing_token: Legacy token value (for backward compatibility)
    
    Returns:
        The API key to use, or None if not configured
    """
    # Try new config system first
    new_key = get_mcp_server_api_key(server_name)
    if new_key:
        return new_key
    
    # Fall back to legacy token
    if existing_token:
        return existing_token
    
    return None


def is_auth_enabled(server_name: str) -> bool:
    """Check if authentication is enabled for a server.
    
    Args:
        server_name: Name of the MCP server
    
    Returns:
        True if an API key is configured, False otherwise
    """
    return get_mcp_server_api_key(server_name) is not None


# ---------------------------------------------------------------------------
# Server-specific validators (convenience functions)
# ---------------------------------------------------------------------------

def require_dev_tools_auth(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
) -> str:
    """Dependency for dev_tools server auth."""
    validator = get_api_key_validator("dev_tools")
    return validator(x_api_key, authorization)


def require_skills_auth(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
) -> str:
    """Dependency for skills server auth."""
    validator = get_api_key_validator("skills")
    return validator(x_api_key, authorization)


def require_control_auth(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
) -> str:
    """Dependency for control server auth."""
    validator = get_api_key_validator("control")
    return validator(x_api_key, authorization)


def require_agentic_loop_auth(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
) -> str:
    """Dependency for agentic_loop server auth."""
    validator = get_api_key_validator("agentic_loop")
    return validator(x_api_key, authorization)


def require_copilot_auth(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
) -> str:
    """Dependency for copilot server auth."""
    validator = get_api_key_validator("copilot")
    return validator(x_api_key, authorization)
