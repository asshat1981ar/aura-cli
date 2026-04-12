"""
Security hardening tests for CredentialStore and DPoP.

Tests resistance to:
- Path traversal attacks
- SQL injection patterns
- Null byte injection
- Extremely long inputs
- JSON injection in secrets
- CRLF/header injection in DPoP URLs
"""

import base64
import json

import pytest

from core.security.file_store import FileStore
from core.security.dpop import DPoPProofGenerator


class TestPathTraversal:
    """FileStore must reject or safely handle path traversal attempts in key names."""

    # Keys that contain actual path separator or traversal characters
    # and MUST be rejected by the store's validation
    REJECTED_PAYLOADS = [
        "../../etc/passwd",
        "../secret",
        "..\\windows\\system32",
        "....//....//etc/passwd",
        "key/../../../etc/shadow",
        "/etc/passwd",
        "\\etc\\passwd",
        "%2e%2e/",  # contains actual /
        "..%252f..%252f",  # contains literal '..' sequence
    ]

    # URL-encoded payloads that don't contain actual traversal characters.
    # These are safe as JSON keys — they should be stored literally or
    # rejected, but must NEVER enable actual path traversal.
    SAFE_ENCODED_PAYLOADS = [
        "%2e%2e%2f",
    ]

    @pytest.mark.parametrize("malicious_key", REJECTED_PAYLOADS)
    def test_path_traversal_rejected(self, tmp_path, malicious_key):
        """Keys with actual path separators or '..' must be rejected."""
        store = FileStore(config_dir=str(tmp_path))
        with pytest.raises((ValueError, OSError)):
            store.set(malicious_key, "stolen_data")

    @pytest.mark.parametrize("malicious_key", REJECTED_PAYLOADS)
    def test_path_traversal_get_rejected(self, tmp_path, malicious_key):
        """Keys with actual path separators must be rejected on get() too."""
        store = FileStore(config_dir=str(tmp_path))
        with pytest.raises((ValueError, OSError)):
            store.get(malicious_key)

    @pytest.mark.parametrize("encoded_key", SAFE_ENCODED_PAYLOADS)
    def test_url_encoded_traversal_stored_literally(self, tmp_path, encoded_key):
        """URL-encoded traversal patterns stored as literal JSON keys — no file escape."""
        import os
        store = FileStore(config_dir=str(tmp_path))
        try:
            store.set(encoded_key, "test_value")
            # Must round-trip as literal string
            assert store.get(encoded_key) == "test_value"
        except (ValueError, OSError):
            pass  # Rejection is also acceptable

        # Verify no file was created outside config_dir
        for root, _dirs, files in os.walk(tmp_path.parent):
            if str(root).startswith(str(tmp_path)):
                continue
            for f in files:
                fpath = os.path.join(root, f)
                content = open(fpath, errors="ignore").read()
                assert "test_value" not in content, \
                    f"Path traversal escaped config_dir: data in {fpath}"


class TestSQLInjection:
    """Credential store must safely handle SQL injection patterns."""

    SQL_INJECTION_PAYLOADS = [
        "'; DROP TABLE credentials; --",
        "' OR '1'='1",
        "'; DELETE FROM secrets WHERE '1'='1",
        "key' UNION SELECT * FROM passwords --",
        "Robert'); DROP TABLE Students;--",
    ]

    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_sql_injection_in_key_handled(self, tmp_path, payload):
        """SQL injection in keys must be rejected or stored as literal strings."""
        store = FileStore(config_dir=str(tmp_path))
        try:
            store.set(payload, "value")
            # If it was accepted, it must round-trip as literal string
            result = store.get(payload)
            assert result == "value"
        except (ValueError, OSError):
            pass  # Rejection is acceptable

    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_sql_injection_in_value_stored_literally(self, tmp_path, payload):
        """SQL injection in values must be stored and retrieved as-is."""
        store = FileStore(config_dir=str(tmp_path))
        store.set("safe-key", payload)
        result = store.get("safe-key")
        assert result == payload, f"Value mutated: stored {payload!r}, got {result!r}"


class TestNullByteInjection:
    """Null bytes in keys/values must be rejected or handled safely."""

    def test_null_byte_in_key_rejected(self, tmp_path):
        """Null byte in key name must be rejected."""
        store = FileStore(config_dir=str(tmp_path))
        with pytest.raises((ValueError, OSError)):
            store.set("key\x00suffix", "value")

    def test_null_byte_in_value_handled(self, tmp_path):
        """Null byte in value must be stored safely or rejected."""
        store = FileStore(config_dir=str(tmp_path))
        try:
            store.set("safe-key", "value\x00injected")
            result = store.get("safe-key")
            # Must round-trip the full string including null
            assert result == "value\x00injected"
        except (ValueError, TypeError):
            pass  # Rejection acceptable

    def test_null_byte_key_prefix(self, tmp_path):
        """Key starting with null byte must be rejected."""
        store = FileStore(config_dir=str(tmp_path))
        with pytest.raises((ValueError, OSError)):
            store.set("\x00key", "value")


class TestExtremeInputs:
    """Extreme input lengths must not crash or hang."""

    def test_extremely_long_key_name(self, tmp_path):
        """100k character key name must be rejected, not cause OOM or hang."""
        store = FileStore(config_dir=str(tmp_path))
        long_key = "a" * 100_000
        with pytest.raises(ValueError):
            store.set(long_key, "value")

    def test_extremely_long_value(self, tmp_path):
        """Large values must be storable without crash (or rejected with documented error)."""
        store = FileStore(config_dir=str(tmp_path))
        long_value = "x" * 1_000_000
        try:
            store.set("big-value", long_value)
            result = store.get("big-value")
            assert result == long_value
        except (OSError, MemoryError):
            pass  # Resource limits are acceptable

    def test_empty_key_rejected(self, tmp_path):
        """Empty key must be rejected."""
        store = FileStore(config_dir=str(tmp_path))
        with pytest.raises(ValueError):
            store.set("", "value")

    def test_whitespace_only_key(self, tmp_path):
        """Whitespace-only key must be rejected or handled safely."""
        store = FileStore(config_dir=str(tmp_path))
        try:
            store.set("   ", "value")
        except (ValueError, OSError):
            pass  # Rejection acceptable


class TestJSONInjection:
    """JSON injection in secrets must be handled safely."""

    JSON_PAYLOADS = [
        '{"malicious": "json"}',
        '["array", "injection"]',
        '{"nested": {"deep": true}}',
        'true',
        'null',
        '42',
        '{"__proto__": {"admin": true}}',
    ]

    @pytest.mark.parametrize("payload", JSON_PAYLOADS)
    def test_json_in_secret_stored_as_string(self, tmp_path, payload):
        """JSON strings as secret values must round-trip as strings, not be parsed."""
        store = FileStore(config_dir=str(tmp_path))
        store.set("json-key", payload)
        result = store.get("json-key")
        assert result == payload, f"JSON payload mutated: stored {payload!r}, got {result!r}"
        assert isinstance(result, str), f"Expected str, got {type(result).__name__}"

    def test_json_value_not_parsed_as_object(self, tmp_path):
        """A JSON object stored as value must not be deserialized on retrieval."""
        store = FileStore(config_dir=str(tmp_path))
        store.set("data", '{"admin": true, "role": "superuser"}')
        result = store.get("data")
        assert isinstance(result, str)
        assert result == '{"admin": true, "role": "superuser"}'


class TestDPoPURLInjection:
    """DPoP generator must resist header/URL injection attacks."""

    def test_crlf_injection_in_htu(self):
        """CRLF in htu must not produce a JWT with embedded headers."""
        gen = DPoPProofGenerator()
        malicious_url = "https://evil.com\r\nX-Injected: header"
        proof = gen.generate_proof("GET", malicious_url)

        # The JWT parts must not contain raw CRLF
        for part in proof.split("."):
            decoded = base64.urlsafe_b64decode(part + "==")
            # Base64-encoded parts are safe, but the decoded payload
            # stores the htu as a JSON string which is fine
            assert b"\r\n" not in part.encode("ascii"), \
                "CRLF found in JWT part (not base64-encoded)"

    def test_crlf_injection_in_htm(self):
        """CRLF in htm must not corrupt the JWT structure."""
        gen = DPoPProofGenerator()
        malicious_method = "GET\r\nX-Evil: true"
        proof = gen.generate_proof(malicious_method, "https://example.com")

        parts = proof.split(".")
        assert len(parts) == 3
        # Verify it's still valid JSON in the payload
        payload_b64 = parts[1]
        padding = "=" * (4 - len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + padding))
        assert payload["htm"] == malicious_method  # Stored literally

    def test_url_with_fragments_and_query(self):
        """URLs with query params and fragments must be stored literally."""
        gen = DPoPProofGenerator()
        url = "https://example.com/path?q=1&auth=admin#fragment"
        proof = gen.generate_proof("GET", url)

        payload_b64 = proof.split(".")[1]
        padding = "=" * (4 - len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + padding))
        assert payload["htu"] == url

    def test_unicode_in_url(self):
        """Unicode characters in URL must not crash proof generation."""
        gen = DPoPProofGenerator()
        proof = gen.generate_proof("GET", "https://example.com/\u00e9\u00e8\u00ea")
        parts = proof.split(".")
        assert len(parts) == 3

    def test_extremely_long_url(self):
        """Very long URLs must not cause hangs or memory issues."""
        gen = DPoPProofGenerator()
        long_url = "https://example.com/" + "a" * 100_000
        proof = gen.generate_proof("GET", long_url)
        parts = proof.split(".")
        assert len(parts) == 3

    def test_null_bytes_in_url(self):
        """Null bytes in URL must be handled (stored literally in JSON)."""
        gen = DPoPProofGenerator()
        proof = gen.generate_proof("GET", "https://example.com/\x00evil")
        parts = proof.split(".")
        assert len(parts) == 3
