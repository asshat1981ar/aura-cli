"""Cryptographic operations for config encryption."""

from .models import EncryptedValue, EncryptionAlgorithm


class CryptoEngine:
    """Encrypt and decrypt values using configured algorithm."""

    def __init__(self, key: bytes, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_GCM):
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes for AES-256")
        self.key = key
        self.algorithm = algorithm

    def encrypt(self, plaintext: str) -> EncryptedValue:
        """Encrypt a plaintext string."""
        if self.algorithm == EncryptionAlgorithm.AES_GCM:
            return self._encrypt_aes_gcm(plaintext)
        elif self.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return self._encrypt_chacha20(plaintext)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def decrypt(self, encrypted: EncryptedValue) -> str:
        """Decrypt an encrypted value."""
        if encrypted.algorithm == EncryptionAlgorithm.AES_GCM:
            return self._decrypt_aes_gcm(encrypted)
        elif encrypted.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return self._decrypt_chacha20(encrypted)
        else:
            raise ValueError(f"Unsupported algorithm: {encrypted.algorithm}")

    def _encrypt_aes_gcm(self, plaintext: str) -> EncryptedValue:
        """Encrypt using AES-256-GCM."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        aesgcm = AESGCM(self.key)
        nonce = self._generate_nonce(12)
        ciphertext_with_tag = aesgcm.encrypt(
            nonce,
            plaintext.encode(),
            None,
        )

        # AES-GCM appends 16-byte tag to ciphertext
        ciphertext = ciphertext_with_tag[:-16]
        tag = ciphertext_with_tag[-16:]

        return EncryptedValue(
            ciphertext=ciphertext,
            nonce=nonce,
            tag=tag,
            algorithm=EncryptionAlgorithm.AES_GCM,
        )

    def _decrypt_aes_gcm(self, encrypted: EncryptedValue) -> str:
        """Decrypt using AES-256-GCM."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        aesgcm = AESGCM(self.key)
        # Reconstruct ciphertext with tag
        ciphertext_with_tag = encrypted.ciphertext + encrypted.tag

        plaintext = aesgcm.decrypt(
            encrypted.nonce,
            ciphertext_with_tag,
            None,
        )

        return plaintext.decode()

    def _encrypt_chacha20(self, plaintext: str) -> EncryptedValue:
        """Encrypt using ChaCha20-Poly1305."""
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

        chacha = ChaCha20Poly1305(self.key)
        nonce = self._generate_nonce(12)
        ciphertext_with_tag = chacha.encrypt(
            nonce,
            plaintext.encode(),
            None,
        )

        # ChaCha20-Poly1305 appends 16-byte tag
        ciphertext = ciphertext_with_tag[:-16]
        tag = ciphertext_with_tag[-16:]

        return EncryptedValue(
            ciphertext=ciphertext,
            nonce=nonce,
            tag=tag,
            algorithm=EncryptionAlgorithm.CHACHA20_POLY1305,
        )

    def _decrypt_chacha20(self, encrypted: EncryptedValue) -> str:
        """Decrypt using ChaCha20-Poly1305."""
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

        chacha = ChaCha20Poly1305(self.key)
        ciphertext_with_tag = encrypted.ciphertext + encrypted.tag

        plaintext = chacha.decrypt(
            encrypted.nonce,
            ciphertext_with_tag,
            None,
        )

        return plaintext.decode()

    def _generate_nonce(self, length: int) -> bytes:
        """Generate a random nonce."""
        import secrets

        return secrets.token_bytes(length)
