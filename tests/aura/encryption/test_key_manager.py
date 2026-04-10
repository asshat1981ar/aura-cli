"""Tests for key manager."""

import base64
import os
import pytest
from pathlib import Path

from aura.encryption.key_manager import KeyManager
from aura.encryption.models import EncryptionConfig, KeySource


class TestKeyManager:
    @pytest.fixture
    def temp_key_file(self, tmp_path):
        key_file = tmp_path / "test_key.txt"
        key_file.write_text("test_encryption_key_32bytes_long!!")
        return str(key_file)
    
    def test_get_key_from_env(self, monkeypatch):
        monkeypatch.setenv("AURA_ENCRYPTION_KEY", "test_encryption_key_32bytes_long!!")
        
        config = EncryptionConfig(key_source=KeySource.ENV_VAR)
        manager = KeyManager(config)
        key = manager.get_key()
        
        assert len(key) == 32
    
    def test_get_key_from_env_not_found(self):
        config = EncryptionConfig(
            key_source=KeySource.ENV_VAR,
            key_env_var="NONEXISTENT_KEY",
        )
        manager = KeyManager(config)
        
        with pytest.raises(KeyError):
            manager.get_key()
    
    def test_get_key_from_file(self, temp_key_file):
        config = EncryptionConfig(
            key_source=KeySource.FILE,
            key_file=temp_key_file,
        )
        manager = KeyManager(config)
        key = manager.get_key()
        
        assert len(key) == 32
    
    def test_get_key_from_file_not_found(self):
        config = EncryptionConfig(
            key_source=KeySource.FILE,
            key_file="/nonexistent/path/key.txt",
        )
        manager = KeyManager(config)
        
        with pytest.raises(FileNotFoundError):
            manager.get_key()
    
    def test_normalize_key_from_string(self):
        config = EncryptionConfig()
        manager = KeyManager(config)
        
        key = manager._normalize_key("short_key")
        assert len(key) == 32
    
    def test_normalize_key_from_base64(self):
        config = EncryptionConfig()
        manager = KeyManager(config)
        
        # Generate a 32-byte base64 encoded key
        original_key = base64.b64encode(b"x" * 32).decode()
        key = manager._normalize_key(original_key)
        
        assert len(key) == 32
        assert key == b"x" * 32
    
    def test_generate_key(self):
        manager = KeyManager()
        key = manager.generate_key()
        
        # Should be base64 encoded
        decoded = base64.b64decode(key)
        assert len(decoded) == 32
    
    def test_save_key_to_file(self, tmp_path):
        manager = KeyManager()
        key = manager.generate_key()
        key_file = tmp_path / "new_key.txt"
        
        manager.save_key_to_file(key, str(key_file))
        
        assert key_file.exists()
        assert key_file.read_text() == key
        # Check permissions are restrictive
        stat = key_file.stat()
        assert stat.st_mode & 0o777 == 0o600
