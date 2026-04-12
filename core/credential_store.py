"""
Secure Credential Store for AURA CLI

Provides OS keyring integration with encrypted file fallback for storing
sensitive credentials like API keys.

Security Issue #427: Migrate API keys from plaintext JSON to OS keyring
"""

import os
import json
import base64
import hashlib
import getpass
from pathlib import Path
from typing import Optional, Dict, Any
from functools import lru_cache

from core.logging_utils import log_json

# Try to import keyring
try:
    import keyring
    from keyring.errors import KeyringError, PasswordSetError

    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False
    keyring = None

# Try to import cryptography for encrypted fallback
try:
    from cryptography.fernet import Fernet

    # Test if Fernet actually works (some envs have broken OpenSSL)
    _test_key = Fernet.generate_key()
    _test_cipher = Fernet(_test_key)
    _test_cipher.encrypt(b"test")
    CRYPTO_AVAILABLE = True
except Exception:
    CRYPTO_AVAILABLE = False
    Fernet = None


class CredentialError(Exception):
    """Raised when credential operations fail."""

    pass


class _SimpleEncryption:
    """
    Simple encryption for fallback when keyring and cryptography are unavailable.

    This uses a basic XOR-based stream cipher with a key derived from machine
    identification. This provides basic obfuscation against casual inspection,
    but should NOT be considered cryptographically secure. It is only used as
    a last resort when better options are unavailable.
    """

    def __init__(self, key: bytes):
        """Initialize with a key."""
        self._key = hashlib.sha256(key).digest()

    @classmethod
    def from_machine_id(cls, machine_id: str, salt: bytes) -> "_SimpleEncryption":
        """Create cipher from machine ID and salt."""
        password = f"{machine_id}:{getpass.getuser()}".encode()
        key_material = hashlib.pbkdf2_hmac("sha256", password, salt, iterations=100000)
        return cls(key_material)

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data using simple stream cipher."""
        # Generate keystream using the key
        key_hash = self._key
        result = bytearray()
        for i, byte in enumerate(data):
            keystream_byte = key_hash[i % len(key_hash)]
            result.append(byte ^ keystream_byte)
        return bytes(result)

    def decrypt(self, data: bytes) -> bytes:
        """Decrypt data (same as encrypt for XOR cipher)."""
        return self.encrypt(data)


class CredentialStore:
    """
    Secure credential storage using OS keyring with encrypted file fallback.

    Priority order for credential retrieval:
    1. Environment variables (highest priority)
    2. OS keyring (if available)
    3. Encrypted fallback file (if keyring unavailable)
    4. Return None (not found)

    Attributes:
        app_name: The application name used as keyring service identifier
        fallback_path: Path to encrypted fallback storage file
        use_keyring: Whether keyring is available and enabled
        use_fallback: Whether encrypted fallback is available and enabled
    """

    # Credential keys that should be stored securely
    SENSITIVE_KEYS = {
        "api_key",
        "openai_api_key",
        "anthropic_api_key",
        "github_token",
        "azure_api_key",
        "aws_access_key_id",
        "aws_secret_access_key",
    }

    def __init__(
        self,
        app_name: str = "aura-cli",
        fallback_path: Optional[Path] = None,
        enable_keyring: bool = True,
        enable_fallback: bool = True,
    ):
        """
        Initialize the credential store.

        Args:
            app_name: Application name for keyring service
            fallback_path: Path for encrypted fallback file
            enable_keyring: Whether to use OS keyring
            enable_fallback: Whether to use encrypted fallback
        """
        self.app_name = app_name
        self.fallback_path = fallback_path or Path.home() / ".aura" / "credentials.enc"
        self.enable_keyring = enable_keyring and KEYRING_AVAILABLE
        self.enable_fallback = enable_fallback and CRYPTO_AVAILABLE

        # Ensure fallback directory exists
        if self.enable_fallback:
            self.fallback_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize fallback cipher if needed
        self._fallback_cipher = None

        log_json(
            "INFO",
            "credential_store_initialized",
            details={
                "keyring_available": self.enable_keyring,
                "fallback_available": self.enable_fallback,
                "fallback_path": str(self.fallback_path),
            },
        )

    def _get_fallback_cipher(self) -> Optional[Any]:
        """Get or create the cipher for fallback encryption."""
        if self._fallback_cipher is not None:
            return self._fallback_cipher

        # Derive key from machine-specific information
        salt_path = self.fallback_path.parent / ".salt"

        # Use or generate salt
        if salt_path.exists():
            salt = salt_path.read_bytes()
        else:
            salt = os.urandom(16)
            salt_path.write_bytes(salt)
            salt_path.chmod(0o600)  # Owner read/write only

        # Use machine-specific identifier combined with user info
        machine_id = self._get_machine_id()

        if Fernet and self.enable_fallback:
            # Use proper Fernet encryption
            password = f"{machine_id}:{getpass.getuser()}".encode()
            key_material = hashlib.pbkdf2_hmac(
                "sha256",
                password,
                salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(key_material)
            self._fallback_cipher = Fernet(key)
        else:
            # Use simple fallback encryption
            self._fallback_cipher = _SimpleEncryption.from_machine_id(machine_id, salt)

        return self._fallback_cipher

    def _get_machine_id(self) -> str:
        """Get a machine-specific identifier for key derivation."""
        # Try multiple sources for machine ID
        try:
            # Linux: machine-id
            machine_id_path = Path("/etc/machine-id")
            if machine_id_path.exists():
                return machine_id_path.read_text().strip()

            # macOS: IOPlatformUUID
            import subprocess

            result = subprocess.run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True,
                text=True,
            )
            for line in result.stdout.split("\n"):
                if "IOPlatformUUID" in line:
                    return line.split('"')[-2]

            # Windows: MachineGuid
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
                return winreg.QueryValueEx(key, "MachineGuid")[0]

        except Exception:
            pass

        # Fallback: use hostname + user
        import socket

        return hashlib.sha256(f"{socket.gethostname()}:{getpass.getuser()}".encode()).hexdigest()[:32]

    def _read_fallback_store(self) -> Dict[str, str]:
        """Read credentials from encrypted fallback file."""
        if not self.fallback_path.exists():
            return {}

        cipher = self._get_fallback_cipher()
        if not cipher:
            return {}

        try:
            encrypted_data = self.fallback_path.read_bytes()
            decrypted_data = cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            log_json("ERROR", "fallback_read_failed", details={"error": str(e)})
            return {}

    def _write_fallback_store(self, data: Dict[str, str]) -> bool:
        """Write credentials to encrypted fallback file."""
        cipher = self._get_fallback_cipher()
        if not cipher:
            return False

        try:
            json_data = json.dumps(data).encode()
            encrypted_data = cipher.encrypt(json_data)
            self.fallback_path.write_bytes(encrypted_data)
            self.fallback_path.chmod(0o600)  # Owner read/write only
            return True
        except Exception as e:
            log_json("ERROR", "fallback_write_failed", details={"error": str(e)})
            return False

    def is_sensitive_key(self, key: str) -> bool:
        """Check if a key should be stored securely."""
        return key in self.SENSITIVE_KEYS or any(suffix in key.lower() for suffix in ["_api_key", "_token", "_secret", "_password"])

    def store(self, key: str, value: str) -> bool:
        """
        Store a credential securely.

        Args:
            key: The credential identifier
            value: The credential value

        Returns:
            True if stored successfully, False otherwise
        """
        if not value:
            return False

        # Try keyring first
        if self.enable_keyring and keyring:
            try:
                keyring.set_password(self.app_name, key, value)
                log_json("INFO", "credential_stored_keyring", details={"key": key})
                return True
            except (KeyringError, PasswordSetError) as e:
                log_json("WARN", "keyring_store_failed", details={"key": key, "error": str(e)})

        # Fallback to encrypted file
        if self.enable_fallback:
            store = self._read_fallback_store()
            store[key] = value
            if self._write_fallback_store(store):
                log_json("INFO", "credential_stored_fallback", details={"key": key})
                return True

        log_json("ERROR", "credential_store_failed", details={"key": key})
        return False

    def retrieve(self, key: str) -> Optional[str]:
        """
        Retrieve a credential.

        Priority:
        1. Environment variable
        2. OS keyring
        3. Encrypted fallback file

        Args:
            key: The credential identifier

        Returns:
            The credential value or None if not found
        """
        # Priority 1: Environment variable
        env_key = f"AURA_{key.upper()}"
        env_value = os.environ.get(env_key) or os.environ.get(key.upper())
        if env_value:
            return env_value

        # Priority 2: OS keyring
        if self.enable_keyring and keyring:
            try:
                value = keyring.get_password(self.app_name, key)
                if value:
                    return value
            except KeyringError as e:
                log_json("WARN", "keyring_retrieve_failed", details={"key": key, "error": str(e)})

        # Priority 3: Encrypted fallback file
        if self.enable_fallback:
            store = self._read_fallback_store()
            if key in store:
                return store[key]

        return None

    def delete(self, key: str) -> bool:
        """
        Delete a credential from all storage backends.

        Args:
            key: The credential identifier

        Returns:
            True if deleted successfully, False otherwise
        """
        deleted = False

        # Delete from keyring
        if self.enable_keyring and keyring:
            try:
                keyring.delete_password(self.app_name, key)
                deleted = True
            except KeyringError:
                pass

        # Delete from fallback
        if self.enable_fallback:
            store = self._read_fallback_store()
            if key in store:
                del store[key]
                if self._write_fallback_store(store):
                    deleted = True

        if deleted:
            log_json("INFO", "credential_deleted", details={"key": key})

        return deleted

    def exists(self, key: str) -> bool:
        """
        Check if a credential exists in secure storage.

        Args:
            key: The credential identifier

        Returns:
            True if the credential exists, False otherwise
        """
        return self.retrieve(key) is not None

    def list_stored_keys(self) -> list:
        """
        List all keys stored in fallback storage.

        Note: Keyring doesn't support listing credentials for security reasons.

        Returns:
            List of keys in fallback storage
        """
        if self.enable_fallback:
            store = self._read_fallback_store()
            return list(store.keys())
        return []

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about the credential store configuration.

        Returns:
            Dictionary with storage configuration info
        """
        return {
            "app_name": self.app_name,
            "keyring_available": self.enable_keyring,
            "fallback_available": self.enable_fallback,
            "fallback_path": str(self.fallback_path),
            "fallback_exists": self.fallback_path.exists(),
            "stored_keys_count": len(self.list_stored_keys()),
        }


# Global instance
@lru_cache()
def get_credential_store() -> CredentialStore:
    """Get the singleton credential store instance."""
    return CredentialStore()


def secure_store(key: str, value: str) -> bool:
    """Convenience function to store a credential."""
    return get_credential_store().store(key, value)


def secure_retrieve(key: str) -> Optional[str]:
    """Convenience function to retrieve a credential."""
    return get_credential_store().retrieve(key)


def secure_delete(key: str) -> bool:
    """Convenience function to delete a credential."""
    return get_credential_store().delete(key)
