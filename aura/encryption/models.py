"""Data models for config encryption."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Set


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""
    AES_GCM = "aes_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"


class KeySource(Enum):
    """Source for encryption key."""
    ENV_VAR = "env_var"
    KEYCHAIN = "keychain"
    FILE = "file"
    PROMPT = "prompt"


@dataclass
class EncryptionConfig:
    """Configuration for encryption operations."""
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_GCM
    key_source: KeySource = KeySource.ENV_VAR
    key_env_var: str = "AURA_ENCRYPTION_KEY"
    key_file: Optional[str] = None
    keychain_service: Optional[str] = None
    keychain_account: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm.value,
            "key_source": self.key_source.value,
            "key_env_var": self.key_env_var,
            "key_file": self.key_file,
            "keychain_service": self.keychain_service,
            "keychain_account": self.keychain_account,
        }


@dataclass
class EncryptedValue:
    """An encrypted value with metadata."""
    ciphertext: bytes
    nonce: bytes
    tag: bytes
    algorithm: EncryptionAlgorithm
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        import base64
        return {
            "ciphertext": base64.b64encode(self.ciphertext).decode(),
            "nonce": base64.b64encode(self.nonce).decode(),
            "tag": base64.b64encode(self.tag).decode(),
            "algorithm": self.algorithm.value,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EncryptedValue":
        import base64
        return cls(
            ciphertext=base64.b64decode(data["ciphertext"]),
            nonce=base64.b64decode(data["nonce"]),
            tag=base64.b64decode(data["tag"]),
            algorithm=EncryptionAlgorithm(data["algorithm"]),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class EncryptedConfig:
    """Configuration with encrypted fields."""
    data: dict
    encrypted_fields: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> dict:
        return {
            "data": self.data,
            "encrypted_fields": list(self.encrypted_fields),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EncryptedConfig":
        return cls(
            data=data.get("data", {}),
            encrypted_fields=set(data.get("encrypted_fields", [])),
        )
