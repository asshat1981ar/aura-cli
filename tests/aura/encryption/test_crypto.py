"""Tests for crypto engine."""

import pytest

from aura.encryption.crypto import CryptoEngine
from aura.encryption.models import EncryptionAlgorithm


class TestCryptoEngine:
    @pytest.fixture
    def key(self):
        return b"x" * 32
    
    @pytest.fixture
    def engine(self, key):
        return CryptoEngine(key, EncryptionAlgorithm.AES_GCM)
    
    def test_invalid_key_length(self):
        with pytest.raises(ValueError, match="Key must be 32 bytes"):
            CryptoEngine(b"too_short")
    
    def test_encrypt_aes_gcm(self, engine):
        encrypted = engine.encrypt("secret message")
        
        assert encrypted.ciphertext is not None
        assert encrypted.nonce is not None
        assert encrypted.tag is not None
        assert encrypted.algorithm == EncryptionAlgorithm.AES_GCM
    
    def test_decrypt_aes_gcm(self, engine):
        original = "secret message"
        encrypted = engine.encrypt(original)
        decrypted = engine.decrypt(encrypted)
        
        assert decrypted == original
    
    def test_encrypt_decrypt_different_messages(self, engine):
        messages = [
            "short",
            "a much longer message with lots of text",
            "message with unicode: 🎉 émojis",
            "",
        ]
        
        for message in messages:
            encrypted = engine.encrypt(message)
            decrypted = engine.decrypt(encrypted)
            assert decrypted == message
    
    def test_different_nonces_each_encryption(self, engine):
        encrypted1 = engine.encrypt("same message")
        encrypted2 = engine.encrypt("same message")
        
        # Same plaintext should produce different ciphertext due to different nonces
        assert encrypted1.nonce != encrypted2.nonce
        assert encrypted1.ciphertext != encrypted2.ciphertext
    
    def test_encrypt_chacha20(self, key):
        engine = CryptoEngine(key, EncryptionAlgorithm.CHACHA20_POLY1305)
        encrypted = engine.encrypt("secret message")
        
        assert encrypted.ciphertext is not None
        assert encrypted.nonce is not None
        assert encrypted.tag is not None
        assert encrypted.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305
    
    def test_decrypt_chacha20(self, key):
        engine = CryptoEngine(key, EncryptionAlgorithm.CHACHA20_POLY1305)
        original = "secret message"
        encrypted = engine.encrypt(original)
        decrypted = engine.decrypt(encrypted)
        
        assert decrypted == original
    
    def test_decrypt_wrong_algorithm(self, engine, key):
        encrypted = engine.encrypt("secret")
        
        # Change algorithm
        encrypted.algorithm = EncryptionAlgorithm.CHACHA20_POLY1305
        
        # Should fail to decrypt with wrong algorithm
        chacha_engine = CryptoEngine(key, EncryptionAlgorithm.CHACHA20_POLY1305)
        with pytest.raises(Exception):
            chacha_engine.decrypt(encrypted)
    
    def test_decrypt_with_wrong_key(self):
        engine1 = CryptoEngine(b"1" * 32, EncryptionAlgorithm.AES_GCM)
        engine2 = CryptoEngine(b"2" * 32, EncryptionAlgorithm.AES_GCM)
        
        encrypted = engine1.encrypt("secret")
        
        with pytest.raises(Exception):
            engine2.decrypt(encrypted)
