"""
Golden file (snapshot) tests for DPoP proof JWT structure.
Verifies that the JWT structure never drifts unexpectedly across refactors.

Update snapshots with: pytest --snapshot-update
"""

import base64
import json

import pytest

from core.security.dpop import DPoPProofGenerator


def decode_b64url(s: str) -> dict:
    padding = "=" * (4 - len(s) % 4)
    return json.loads(base64.urlsafe_b64decode(s + padding))


@pytest.fixture
def deterministic_generator(monkeypatch):
    """
    Returns a generator with mocked time and jti for deterministic snapshot output.
    """
    import secrets
    import time

    monkeypatch.setattr(time, "time", lambda: 1712678400.0)  # Fixed timestamp
    monkeypatch.setattr(secrets, "token_urlsafe", lambda n=16: "FIXED_JTI_FOR_SNAPSHOT")
    return DPoPProofGenerator()


class TestDPoPGoldenStructure:

    def test_proof_header_structure(self, deterministic_generator, snapshot):
        """Header fields must match snapshot exactly."""
        proof = deterministic_generator.generate_proof(
            "GET", "https://api.aura.example.com/v1/instances"
        )
        header = decode_b64url(proof.split(".")[0])
        # Snapshot the header (excluding the JWK x/y since EC key is always ephemeral)
        assert {
            "typ": header["typ"],
            "alg": header["alg"],
            "jwk_kty": header["jwk"]["kty"],
            "jwk_crv": header["jwk"]["crv"],
        } == snapshot

    def test_proof_payload_structure_no_token(self, deterministic_generator, snapshot):
        """Payload without access_token must match snapshot structure."""
        proof = deterministic_generator.generate_proof(
            "POST", "https://api.aura.example.com/v1/instances"
        )
        payload = decode_b64url(proof.split(".")[1])
        # Snapshot all fields except jti (mocked but exclude for safety)
        assert {k: v for k, v in payload.items() if k != "jti"} == snapshot

    def test_proof_payload_with_token(self, deterministic_generator, snapshot):
        """Payload with access_token must include ath claim — snapshot its structure."""
        proof = deterministic_generator.generate_proof(
            "DELETE",
            "https://api.aura.example.com/v1/instances/abc-123",
            access_token="test_bearer_token_for_snapshot",
        )
        payload = decode_b64url(proof.split(".")[1])
        assert {k: v for k, v in payload.items() if k != "jti"} == snapshot

    def test_proof_is_three_parts(self, deterministic_generator, snapshot):
        """A DPoP proof must always be exactly 3 base64url parts separated by dots."""
        proof = deterministic_generator.generate_proof("GET", "https://api.example.com")
        structure = {
            "parts": len(proof.split(".")),
            "separator": ".",
            "pattern": "base64url.base64url.base64url",
        }
        assert structure == snapshot
