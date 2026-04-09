"""Tests for RFC 9449 DPoP implementation."""
import base64
import hashlib
import json
import time
from unittest.mock import MagicMock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric import ec, utils as ec_utils
from cryptography.hazmat.primitives import hashes

from core.security.dpop import (
    DPoPProofGenerator,
    get_dpop_generator,
    reset_dpop_generator,
)
from core.security.http_client import DPoPSession, attach_dpop_headers


def _b64url_decode(s: str) -> bytes:
    """Decode a base64url string (without padding)."""
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def _parse_jwt(token: str) -> tuple[dict, dict, bytes]:
    """Split a compact JWT into (header, payload, signature_bytes)."""
    parts = token.split(".")
    assert len(parts) == 3, f"Expected 3-part JWT, got {len(parts)} parts"
    header = json.loads(_b64url_decode(parts[0]))
    payload = json.loads(_b64url_decode(parts[1]))
    sig = _b64url_decode(parts[2])
    return header, payload, sig


class TestDPoPProofGenerator:
    """Test the core DPoP proof generator."""

    def setup_method(self):
        self.gen = DPoPProofGenerator()

    def test_generates_valid_three_part_jwt(self):
        proof = self.gen.generate_proof("POST", "https://api.example.com/v1/foo")
        parts = proof.split(".")
        assert len(parts) == 3
        # Each part should be non-empty base64url
        for part in parts:
            assert len(part) > 0

    def test_header_fields(self):
        proof = self.gen.generate_proof("GET", "https://example.com/resource")
        header, _, _ = _parse_jwt(proof)
        assert header["typ"] == "dpop+jwt"
        assert header["alg"] == "ES256"
        jwk = header["jwk"]
        assert jwk["kty"] == "EC"
        assert jwk["crv"] == "P-256"
        assert "x" in jwk
        assert "y" in jwk

    def test_payload_fields(self):
        proof = self.gen.generate_proof("POST", "https://api.example.com/call")
        _, payload, _ = _parse_jwt(proof)
        assert payload["jti"]  # non-empty
        assert payload["htm"] == "POST"
        assert payload["htu"] == "https://api.example.com/call"
        # iat should be a recent unix timestamp (within 5 seconds)
        assert abs(payload["iat"] - int(time.time())) < 5

    def test_htm_uppercased(self):
        proof = self.gen.generate_proof("post", "https://example.com/x")
        _, payload, _ = _parse_jwt(proof)
        assert payload["htm"] == "POST"

    def test_ath_present_when_token_provided(self):
        token = "eyJhbGciOiJSUzI1NiJ9.test.payload"
        proof = self.gen.generate_proof("POST", "https://api.example.com/v1", access_token=token)
        _, payload, _ = _parse_jwt(proof)
        assert "ath" in payload
        # Verify ath == base64url(SHA256(token))
        expected = (
            base64.urlsafe_b64encode(hashlib.sha256(token.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        assert payload["ath"] == expected

    def test_no_ath_without_token(self):
        proof = self.gen.generate_proof("GET", "https://example.com/x")
        _, payload, _ = _parse_jwt(proof)
        assert "ath" not in payload

    def test_unique_jti_per_proof(self):
        proof1 = self.gen.generate_proof("GET", "https://example.com/a")
        proof2 = self.gen.generate_proof("GET", "https://example.com/a")
        _, p1, _ = _parse_jwt(proof1)
        _, p2, _ = _parse_jwt(proof2)
        assert p1["jti"] != p2["jti"]

    def test_signature_verifies(self):
        proof = self.gen.generate_proof("POST", "https://example.com/test")
        parts = proof.split(".")
        signing_input = f"{parts[0]}.{parts[1]}".encode()
        sig_bytes = _b64url_decode(parts[2])

        # Convert raw R||S back to DER for verification
        assert len(sig_bytes) == 64
        r = int.from_bytes(sig_bytes[:32], "big")
        s = int.from_bytes(sig_bytes[32:], "big")
        der_sig = ec_utils.encode_dss_signature(r, s)

        # Should not raise
        self.gen._public_key.verify(der_sig, signing_input, ec.ECDSA(hashes.SHA256()))

    def test_public_jwk_property(self):
        jwk = self.gen.public_jwk
        assert jwk["kty"] == "EC"
        assert jwk["crv"] == "P-256"
        # Should return a copy, not the internal dict
        jwk["extra"] = "test"
        assert "extra" not in self.gen.public_jwk


class TestGetDPoPGenerator:
    """Test the process-scoped singleton."""

    def setup_method(self):
        reset_dpop_generator()

    def teardown_method(self):
        reset_dpop_generator()

    def test_returns_same_instance(self):
        g1 = get_dpop_generator()
        g2 = get_dpop_generator()
        assert g1 is g2

    def test_reset_creates_new_instance(self):
        g1 = get_dpop_generator()
        reset_dpop_generator()
        g2 = get_dpop_generator()
        assert g1 is not g2


class TestDPoPSession:
    """Test the requests.Session wrapper."""

    def setup_method(self):
        reset_dpop_generator()

    def teardown_method(self):
        reset_dpop_generator()

    def test_attaches_auth_and_dpop_headers(self):
        mock_session = MagicMock()
        mock_session.request.return_value = MagicMock(status_code=200)

        session = DPoPSession(session=mock_session)
        session.post("https://api.example.com/v1/chat", access_token="my-token")

        call_args = mock_session.request.call_args
        headers = call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer my-token"
        assert "DPoP" in headers
        # Verify the DPoP value is a valid JWT
        parts = headers["DPoP"].split(".")
        assert len(parts) == 3

    def test_no_dpop_header_without_token(self):
        mock_session = MagicMock()
        mock_session.request.return_value = MagicMock(status_code=200)

        session = DPoPSession(session=mock_session)
        session.get("https://api.example.com/health")

        call_args = mock_session.request.call_args
        headers = call_args.kwargs["headers"]
        assert "Authorization" not in headers
        assert "DPoP" not in headers

    def test_convenience_methods(self):
        mock_session = MagicMock()
        mock_session.request.return_value = MagicMock(status_code=200)
        session = DPoPSession(session=mock_session)

        session.get("https://example.com/a", access_token="t")
        assert mock_session.request.call_args[0] == ("GET", "https://example.com/a")

        session.put("https://example.com/b", access_token="t")
        assert mock_session.request.call_args[0] == ("PUT", "https://example.com/b")

        session.patch("https://example.com/c", access_token="t")
        assert mock_session.request.call_args[0] == ("PATCH", "https://example.com/c")

        session.delete("https://example.com/d", access_token="t")
        assert mock_session.request.call_args[0] == ("DELETE", "https://example.com/d")


class TestAttachDPoPHeaders:
    """Test the standalone header attachment helper."""

    def setup_method(self):
        reset_dpop_generator()

    def teardown_method(self):
        reset_dpop_generator()

    def test_adds_dpop_header_to_dict(self):
        headers = {"Authorization": "Bearer abc123"}
        result = attach_dpop_headers(headers, "POST", "https://api.example.com/v1/foo", access_token="abc123")
        assert "DPoP" in result
        assert result is headers  # modifies in-place

    def test_strips_query_fragment_from_htu(self):
        headers = {}
        attach_dpop_headers(headers, "GET", "https://example.com/path?q=1#frag")
        proof = headers["DPoP"]
        _, payload, _ = _parse_jwt(proof)
        assert payload["htu"] == "https://example.com/path"
