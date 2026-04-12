"""Tests for encryption models."""

import pytest
from datetime import datetime

from aura.encryption.models import (
    EncryptedConfig,
    EncryptedValue,
    EncryptionAlgorithm,
    EncryptionConfig,
    KeySource,
)


class TestEncryptionAlgorithm:
    def test_algorithm_values(self):
        assert EncryptionAlgorithm.AES_GCM.value == "aes_gcm"
        assert EncryptionAlgorithm.CHACHA20_POLY1305.value == "chacha20_poly1305"


class TestKeySource:
    def test_key_source_values(self):
        assert KeySource.ENV_VAR.value == "env_var"
        assert KeySource.KEYCHAIN.value == "keychain"
        assert KeySource.FILE.value == "file"
        assert KeySource.PROMPT.value == "prompt"


class TestEncryptionConfig:
    def test_default_values(self):
        config = EncryptionConfig()
        
        assert config.algorithm == EncryptionAlgorithm.AES_GCM
        assert config.key_source == KeySource.ENV_VAR
        assert config.key_env_var == "AURA_ENCRYPTION_KEY"
        assert config.key_file is None
        assert config.keychain_service is None
        assert config.keychain_account is None
    
    def test_custom_values(self):
        config = EncryptionConfig(
            algorithm=EncryptionAlgorithm.CHACHA20_POLY1305,
            key_source=KeySource.FILE,
            key_env_var="CUSTOM_KEY",
            key_file="/path/to/key",
            keychain_service="my-service",
            keychain_account="my-account",
        )
        
        assert config.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305
        assert config.key_source == KeySource.FILE
        assert config.key_env_var == "CUSTOM_KEY"
        assert config.key_file == "/path/to/key"
        assert config.keychain_service == "my-service"
        assert config.keychain_account == "my-account"
    
    def test_to_dict(self):
        config = EncryptionConfig(
            algorithm=EncryptionAlgorithm.CHACHA20_POLY1305,
            key_source=KeySource.FILE,
            key_file="/path/to/key",
        )
        
        data = config.to_dict()
        
        assert data["algorithm"] == "chacha20_poly1305"
        assert data["key_source"] == "file"
        assert data["key_file"] == "/path/to/key"


class TestEncryptedValue:
    def test_creation(self):
        value = EncryptedValue(
            ciphertext=b"encrypted_data",
            nonce=b"nonce123",
            tag=b"tag456",
            algorithm=EncryptionAlgorithm.AES_GCM,
        )
        
        assert value.ciphertext == b"encrypted_data"
        assert value.nonce == b"nonce123"
        assert value.tag == b"tag456"
        assert value.algorithm == EncryptionAlgorithm.AES_GCM
        assert isinstance(value.created_at, datetime)
    
    def test_to_dict(self):
        value = EncryptedValue(
            ciphertext=b"encrypted_data",
            nonce=b"nonce123",
            tag=b"tag456",
            algorithm=EncryptionAlgorithm.AES_GCM,
        )
        
        data = value.to_dict()
        
        assert data["ciphertext"] == "ZW5jcnlwdGVkX2RhdGE="
        assert data["nonce"] == "bm9uY2UxMjM="
        assert data["tag"] == "dGFnNDU2"
        assert data["algorithm"] == "aes_gcm"
        assert "created_at" in data
    
    def test_from_dict(self):
        data = {
            "ciphertext": "ZW5jcnlwdGVkX2RhdGE=",
            "nonce": "bm9uY2UxMjM=",
            "tag": "dGFnNDU2",
            "algorithm": "aes_gcm",
            "created_at": "2024-01-01T00:00:00",
        }
        
        value = EncryptedValue.from_dict(data)
        
        assert value.ciphertext == b"encrypted_data"
        assert value.nonce == b"nonce123"
        assert value.tag == b"tag456"
        assert value.algorithm == EncryptionAlgorithm.AES_GCM


class TestEncryptedConfig:
    def test_default_creation(self):
        config = EncryptedConfig(data={})
        
        assert config.data == {}
        assert config.encrypted_fields == set()
    
    def test_with_encrypted_fields(self):
        config = EncryptedConfig(
            data={"api_key": "secret123"},
            encrypted_fields={"api_key"},
        )
        
        assert config.data == {"api_key": "secret123"}
        assert config.encrypted_fields == {"api_key"}
    
    def test_to_dict(self):
        config = EncryptedConfig(
            data={"api_key": "secret123"},
            encrypted_fields={"api_key"},
        )
        
        data = config.to_dict()
        
        assert data["data"] == {"api_key": "secret123"}
        assert data["encrypted_fields"] == ["api_key"]
    
    def test_from_dict(self):
        data = {
            "data": {"api_key": "secret123"},
            "encrypted_fields": ["api_key", "password"],
        }
        
        config = EncryptedConfig.from_dict(data)
        
        assert config.data == {"api_key": "secret123"}
        assert config.encrypted_fields == {"api_key", "password"}
