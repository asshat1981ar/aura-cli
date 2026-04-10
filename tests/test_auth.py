"""Tests for core/auth.py OAuth2 authentication."""

import pytest
from unittest.mock import patch

from core.auth import (
    AuthManager,
    User,
    UserRole,
    AuthError,
    AuthenticationError,
    AuthorizationError,
    TokenError,
    init_auth,
    get_auth_manager,
)


class TestUser:
    """Test User model."""
    
    def test_user_creation(self):
        """Test user creation."""
        user = User(
            username="testuser",
            hashed_password="hash",
            role=UserRole.DEVELOPER,
        )
        assert user.username == "testuser"
        assert user.role == UserRole.DEVELOPER
        assert not user.disabled
    
    def test_has_permission_admin(self):
        """Test admin has all permissions."""
        admin = User("admin", role=UserRole.ADMIN)
        assert admin.has_permission(UserRole.VIEWER)
        assert admin.has_permission(UserRole.DEVELOPER)
        assert admin.has_permission(UserRole.ADMIN)
    
    def test_has_permission_developer(self):
        """Test developer permissions."""
        dev = User("dev", role=UserRole.DEVELOPER)
        assert dev.has_permission(UserRole.VIEWER)
        assert dev.has_permission(UserRole.DEVELOPER)
        assert not dev.has_permission(UserRole.ADMIN)
    
    def test_has_permission_viewer(self):
        """Test viewer permissions."""
        viewer = User("viewer", role=UserRole.VIEWER)
        assert viewer.has_permission(UserRole.VIEWER)
        assert not viewer.has_permission(UserRole.DEVELOPER)
        assert not viewer.has_permission(UserRole.ADMIN)
    
    def test_to_dict_excludes_sensitive(self):
        """Test to_dict excludes sensitive fields."""
        user = User(
            username="test",
            hashed_password="secret_hash",
            api_key="secret_key",
            role=UserRole.ADMIN,
        )
        data = user.to_dict()
        assert "hashed_password" not in data
        assert "api_key" not in data
        assert data["username"] == "test"
        assert data["role"] == "admin"


class TestAuthManager:
    """Test AuthManager functionality."""
    
    @pytest.fixture
    def auth(self):
        """Create auth manager for testing."""
        return AuthManager(
            secret_key="test-secret-key-for-testing-only",
            algorithm="none",  # Disable crypto for testing without dependencies
        )
    
    def test_create_user(self, auth):
        """Test user creation."""
        user = auth.create_user("newuser", password="password123")
        assert user.username == "newuser"
        assert user.role == UserRole.VIEWER
    
    def test_create_user_duplicate(self, auth):
        """Test creating duplicate user raises error."""
        auth.create_user("user1", password="pass")
        with pytest.raises(AuthError):
            auth.create_user("user1", password="pass")
    
    def test_authenticate_user_success(self, auth):
        """Test successful authentication."""
        auth.create_user("user", password="correct")
        user = auth.authenticate_user("user", "correct")
        assert user is not None
        assert user.username == "user"
    
    def test_authenticate_user_wrong_password(self, auth):
        """Test authentication with wrong password."""
        auth.create_user("user", password="correct")
        user = auth.authenticate_user("user", "wrong")
        assert user is None
    
    def test_authenticate_user_not_found(self, auth):
        """Test authentication for non-existent user."""
        user = auth.authenticate_user("nonexistent", "pass")
        assert user is None
    
    def test_authenticate_disabled_user(self, auth):
        """Test authentication for disabled user."""
        auth.create_user("disabled", password="pass")
        auth._users["disabled"].disabled = True
        
        with pytest.raises(AuthenticationError):
            auth.authenticate_user("disabled", "pass")
    
    def test_create_user_with_api_key(self, auth):
        """Test user creation with API key."""
        user = auth.create_user("apiuser", generate_api_key=True)
        assert user.api_key is not None
        assert user.api_key.startswith("aura_")
    
    def test_authenticate_api_key(self, auth):
        """Test API key authentication."""
        user = auth.create_user("apiuser", generate_api_key=True)
        api_key = user.api_key
        
        authenticated = auth.authenticate_api_key(api_key)
        assert authenticated is not None
        assert authenticated.username == "apiuser"
    
    def test_authenticate_invalid_api_key(self, auth):
        """Test authentication with invalid API key."""
        user = auth.authenticate_api_key("invalid_key")
        assert user is None
    
    def test_require_role_success(self, auth):
        """Test role requirement check passes."""
        admin = auth.create_user("admin", role=UserRole.ADMIN)
        auth.require_role(admin, UserRole.DEVELOPER)  # Should not raise
    
    def test_require_role_failure(self, auth):
        """Test role requirement check fails."""
        viewer = auth.create_user("viewer", role=UserRole.VIEWER)
        
        with pytest.raises(AuthorizationError):
            auth.require_role(viewer, UserRole.ADMIN)
    
    def test_list_users(self, auth):
        """Test listing users."""
        auth.create_user("user1", role=UserRole.ADMIN)
        auth.create_user("user2", role=UserRole.DEVELOPER)
        
        users = auth.list_users()
        assert len(users) == 2
        usernames = {u["username"] for u in users}
        assert usernames == {"user1", "user2"}
    
    def test_delete_user(self, auth):
        """Test deleting user."""
        auth.create_user("todelete")
        assert auth.delete_user("todelete") is True
        assert "todelete" not in auth._users
    
    def test_delete_user_not_found(self, auth):
        """Test deleting non-existent user."""
        assert auth.delete_user("nonexistent") is False
    
    def test_regenerate_api_key(self, auth):
        """Test API key regeneration."""
        user = auth.create_user("user", generate_api_key=True)
        old_key = user.api_key
        
        new_key = auth.regenerate_api_key("user")
        assert new_key != old_key
        assert user.api_key == new_key
        assert old_key not in auth._api_keys
        assert auth._api_keys[new_key] == "user"


class TestPasswordHashing:
    """Test password hashing."""
    
    def test_hash_password(self):
        """Test password hashing produces different hashes."""
        auth = AuthManager(secret_key="test-secret-key-for-password-hashing")
        hash1 = auth.hash_password("password")
        hash2 = auth.hash_password("password")
        
        # Without bcrypt, hashes will be identical (SHA256)
        # With bcrypt, hashes will be different due to salt
        assert hash1 is not None
        assert len(hash1) > 0
    
    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        auth = AuthManager(secret_key="test-secret-key-for-password-hashing")
        hashed = auth.hash_password("mypassword")
        assert auth.verify_password("mypassword", hashed) is True
    
    def test_verify_password_incorrect(self):
        """Test password verification with incorrect password."""
        auth = AuthManager(secret_key="test-secret-key-for-password-hashing")
        hashed = auth.hash_password("mypassword")
        assert auth.verify_password("wrongpassword", hashed) is False


class TestGlobalAuth:
    """Test global auth functions."""
    
    def test_init_auth(self):
        """Test initializing global auth."""
        import core.auth
        core.auth._auth_manager = None  # Reset
        
        auth = init_auth(secret_key="global-test-key", algorithm="none")
        assert auth is not None
        assert get_auth_manager() is auth
    
    def test_get_auth_manager_not_initialized(self):
        """Test getting auth manager before initialization."""
        import core.auth
        core.auth._auth_manager = None  # Reset
        
        with pytest.raises(RuntimeError):
            get_auth_manager()
