"""OAuth2 authentication and authorization for AURA CLI.

Provides JWT-based authentication with:
- OAuth2 password flow for CLI/API access
- API key authentication for service-to-service
- Role-based access control (RBAC)
- Token refresh and revocation
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from core.logging_utils import log_json

_ALLOWED_ALGORITHMS = {"HS256"}


def _default_auth_db_path() -> Path:
    custom = os.environ.get("AURA_AUTH_DB_PATH")
    if custom:
        return Path(custom)
    xdg_data = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg_data) if xdg_data else Path.home() / ".local" / "share"
    return base / "aura" / "auth.db"


# Optional JWT support
try:
    from jose import JWTError, jwt

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    jwt = None
    JWTError = Exception

# Optional password hashing
try:
    from passlib.context import CryptContext

    PWD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")
except ImportError:
    PWD_CONTEXT = None


class AuthError(Exception):
    """Authentication/authorization error."""

    pass


class AuthenticationError(AuthError):
    """Invalid credentials."""

    pass


class AuthorizationError(AuthError):
    """Insufficient permissions."""

    pass


class TokenError(AuthError):
    """Invalid or expired token."""

    pass


class UserRole(str, Enum):
    """User roles for RBAC."""

    ADMIN = "admin"
    DEVELOPER = "developer"
    VIEWER = "viewer"
    SERVICE = "service"


class User:
    """User model for authentication."""

    def __init__(
        self,
        username: str,
        hashed_password: Optional[str] = None,
        role: UserRole = UserRole.VIEWER,
        api_key: Optional[str] = None,
        disabled: bool = False,
        metadata: Optional[dict] = None,
    ):
        self.username = username
        self.hashed_password = hashed_password
        self.role = role
        self.api_key = api_key
        self.disabled = disabled
        self.metadata = metadata or {}

    def has_permission(self, required_role: UserRole) -> bool:
        """Check if user has required role or higher."""
        role_hierarchy = {
            UserRole.VIEWER: 0,
            UserRole.DEVELOPER: 1,
            UserRole.ADMIN: 2,
            UserRole.SERVICE: 3,
        }
        return role_hierarchy.get(self.role, 0) >= role_hierarchy.get(required_role, 0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize user to dict (excludes sensitive fields)."""
        return {
            "username": self.username,
            "role": self.role.value,
            "disabled": self.disabled,
            "metadata": self.metadata,
        }


class AuthManager:
    """Manages authentication and authorization."""

    # Maximum access token lifetime enforced at issue time (24 hours)
    MAX_ACCESS_TOKEN_HOURS: int = 24

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
        db_path: Optional[Any] = None,
    ):
        # Hard block algorithm=none unless running under pytest or explicit test mode
        if algorithm == "none":
            if not (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("AURA_TEST_MODE") == "1"):
                raise ValueError("algorithm='none' is forbidden in production. Set AURA_TEST_MODE=1 only in test environments.")
        elif algorithm not in _ALLOWED_ALGORITHMS:
            raise ValueError(f"Algorithm '{algorithm}' not in allowlist {_ALLOWED_ALGORITHMS}. Only HS256 is supported.")

        self._jwt_available = JWT_AVAILABLE
        if not self._jwt_available and algorithm not in ("none", "dummy"):
            # Allow initialization without JWT for non-token operations
            pass

        # Enforce minimum key length for real algorithms (not test stubs)
        if algorithm not in ("none", "dummy") and len(secret_key.encode()) < 32:
            raise ValueError(f"JWT_SECRET_KEY must be ≥32 bytes (256 bits). Current key is {len(secret_key.encode())} bytes.")

        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self._users: dict[str, User] = {}
        self._api_keys: dict[str, str] = {}  # api_key -> username
        # Legacy in-memory set kept for non-jti revocations; primary store is SQLite
        self._revoked_tokens: set[str] = set()
        resolved_db_path: Path = Path(db_path) if db_path is not None else _default_auth_db_path()
        resolved_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(resolved_db_path)
        self._init_revocation_db()

    def _init_revocation_db(self) -> None:
        """Create the SQLite revocation store if it doesn't exist yet."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS revoked_tokens (
                    jti TEXT PRIMARY KEY,
                    revoked_at TEXT NOT NULL
                )
                """
            )

    def _is_jti_revoked(self, jti: str) -> bool:
        """Return True if *jti* is present in the SQLite revocation table."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=-64000")
                conn.execute("PRAGMA busy_timeout=5000")
                conn.execute("PRAGMA foreign_keys=ON")
                row = conn.execute("SELECT 1 FROM revoked_tokens WHERE jti = ?", (jti,)).fetchone()
                return row is not None
        except sqlite3.Error:
            return False

    def _persist_jti(self, jti: str) -> None:
        """Insert *jti* into the SQLite revocation table."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=-64000")
                conn.execute("PRAGMA busy_timeout=5000")
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute(
                    "INSERT OR IGNORE INTO revoked_tokens (jti, revoked_at) VALUES (?, ?)",
                    (jti, datetime.now(timezone.utc).isoformat()),
                )
        except sqlite3.Error:
            pass  # Degrade gracefully; token will expire naturally

    def hash_password(self, password: str) -> str:
        """Hash a password."""
        if PWD_CONTEXT:
            return PWD_CONTEXT.hash(password)
        # Fallback: simple SHA256 (NOT for production)
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        if PWD_CONTEXT:
            return PWD_CONTEXT.verify(plain_password, hashed_password)
        # Fallback: simple SHA256 (NOT for production)
        return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password

    def create_user(
        self,
        username: str,
        password: Optional[str] = None,
        role: UserRole = UserRole.VIEWER,
        generate_api_key: bool = False,
    ) -> User:
        """Create a new user."""
        if username in self._users:
            raise AuthError(f"User '{username}' already exists")

        hashed_password = self.hash_password(password) if password else None
        api_key = None

        if generate_api_key:
            api_key = self._generate_api_key()

        user = User(
            username=username,
            hashed_password=hashed_password,
            role=role,
            api_key=api_key,
        )
        self._users[username] = user

        if api_key:
            self._api_keys[api_key] = username

        log_json("INFO", "user_created", {"username": username, "role": role.value})
        return user

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user with username and password."""
        user = self._users.get(username)
        if not user:
            return None
        if user.disabled:
            raise AuthenticationError("User account is disabled")
        if not user.hashed_password:
            raise AuthenticationError("User has no password set")
        if not self.verify_password(password, user.hashed_password):
            return None

        log_json("INFO", "user_authenticated", {"username": username})
        return user

    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate using API key."""
        username = self._api_keys.get(api_key)
        if not username:
            return None

        user = self._users.get(username)
        if not user or user.disabled:
            return None

        log_json("INFO", "api_key_authenticated", {"username": username})
        return user

    def create_access_token(
        self,
        user: User,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """Create JWT access token for user.

        The lifetime is capped at MAX_ACCESS_TOKEN_HOURS (24 h) regardless of
        what *expires_delta* requests, to prevent unbounded session tokens.
        Algorithm is always the instance algorithm (HS256 by default).
        """
        if not self._jwt_available:
            raise ImportError("JWT not available. Install with: pip install python-jose[cryptography]")

        if expires_delta is None:
            expires_delta = timedelta(minutes=self.access_token_expire_minutes)

        # Hard cap: never exceed MAX_ACCESS_TOKEN_HOURS
        max_delta = timedelta(hours=self.MAX_ACCESS_TOKEN_HOURS)
        if expires_delta > max_delta:
            expires_delta = max_delta

        expire = datetime.now(timezone.utc) + expires_delta

        payload = {
            "sub": user.username,
            "role": user.role.value,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": secrets.token_urlsafe(16),
            "type": "access",
        }

        # Explicitly specify algorithm (HS256) — never allow algorithm:none
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        log_json("INFO", "access_token_created", {"username": user.username})
        return token

    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token."""
        if not self._jwt_available:
            raise ImportError("JWT not available. Install with: pip install python-jose[cryptography]")

        expire = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)

        payload = {
            "sub": user.username,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": secrets.token_urlsafe(16),
            "type": "refresh",
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        log_json("INFO", "refresh_token_created", {"username": user.username})
        return token

    def verify_token(self, token: str, token_type: str = "access") -> dict[str, Any]:
        """Verify and decode a JWT token.

        Decodes with the explicit algorithm list to prevent algorithm-confusion
        attacks (e.g. alg:none), then checks the jti against the SQLite
        revocation store before returning the payload.
        """
        if not self._jwt_available:
            raise ImportError("JWT not available. Install with: pip install python-jose[cryptography]")

        # Legacy in-memory revocation check (kept for backward compat)
        if token in self._revoked_tokens:
            raise TokenError("Token has been revoked")

        try:
            # algorithms= list prevents algorithm-confusion; explicit HS256 only
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            if payload.get("type") != token_type:
                raise TokenError(f"Invalid token type. Expected {token_type}")

            # Check SQLite jti revocation store (survives process restarts)
            jti = payload.get("jti")
            if jti and self._is_jti_revoked(jti):
                raise TokenError("Token has been revoked")

            return payload

        except JWTError as e:
            raise TokenError(f"Invalid token: {e}")

    def get_current_user(self, token: str) -> User:
        """Get user from access token."""
        payload = self.verify_token(token, "access")
        username = payload.get("sub")

        if not username:
            raise TokenError("Token missing subject")

        user = self._users.get(username)
        if not user:
            raise TokenError("User not found")
        if user.disabled:
            raise AuthenticationError("User account is disabled")

        return user

    def revoke_token(self, token: str) -> None:
        """Revoke a token by persisting its jti to the SQLite blocklist.

        Falls back to the in-memory set when the token cannot be decoded
        (e.g. JWT library not available or token already malformed).
        """
        try:
            if self._jwt_available:
                # Decode without verifying expiry so we can still revoke
                # tokens that are about to expire during a logout request.
                payload = jwt.decode(
                    token,
                    self.secret_key,
                    algorithms=[self.algorithm],
                    options={"verify_exp": False},
                )
                jti = payload.get("jti")
                if jti:
                    self._persist_jti(jti)
        except Exception:
            pass  # Fall through to legacy in-memory store
        self._revoked_tokens.add(token)
        log_json("INFO", "token_revoked")

    def refresh_access_token(self, refresh_token: str) -> str:
        """Create new access token from refresh token."""
        payload = self.verify_token(refresh_token, "refresh")
        username = payload.get("sub")

        if not username:
            raise TokenError("Refresh token missing subject")

        user = self._users.get(username)
        if not user or user.disabled:
            raise TokenError("User not found or disabled")

        return self.create_access_token(user)

    def require_role(self, user: User, required_role: UserRole) -> None:
        """Check if user has required role."""
        if not user.has_permission(required_role):
            raise AuthorizationError(f"User '{user.username}' with role '{user.role.value}' does not have required permission '{required_role.value}'")

    def _generate_api_key(self) -> str:
        """Generate a secure API key."""
        return f"aura_{secrets.token_urlsafe(32)}"

    def regenerate_api_key(self, username: str) -> str:
        """Regenerate API key for user."""
        user = self._users.get(username)
        if not user:
            raise AuthError(f"User '{username}' not found")

        # Remove old API key
        if user.api_key:
            self._api_keys.pop(user.api_key, None)

        # Generate new key
        new_key = self._generate_api_key()
        user.api_key = new_key
        self._api_keys[new_key] = username

        log_json("INFO", "api_key_regenerated", {"username": username})
        return new_key

    def list_users(self) -> list[dict[str, Any]]:
        """List all users (without sensitive data)."""
        return [user.to_dict() for user in self._users.values()]

    def delete_user(self, username: str) -> bool:
        """Delete a user."""
        user = self._users.pop(username, None)
        if user:
            if user.api_key:
                self._api_keys.pop(user.api_key, None)
            log_json("INFO", "user_deleted", {"username": username})
            return True
        return False


# Global auth manager instance
_auth_manager: Optional[AuthManager] = None


def init_auth(
    secret_key: str,
    algorithm: str = "HS256",
    access_token_expire_minutes: int = 30,
) -> AuthManager:
    """Initialize global auth manager."""
    global _auth_manager
    _auth_manager = AuthManager(
        secret_key=secret_key,
        algorithm=algorithm,
        access_token_expire_minutes=access_token_expire_minutes,
    )
    return _auth_manager


def get_auth_manager() -> AuthManager:
    """Get global auth manager."""
    if _auth_manager is None:
        raise RuntimeError("Auth manager not initialized. Call init_auth() first.")
    return _auth_manager
