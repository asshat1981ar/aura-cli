"""Unit tests for DPoP proof generator."""

from __future__ import annotations

import json
import re
import time

from core.security.dpop import DPoPProofGenerator, _b64url_decode


def _decode_jwt_parts(token: str) -> tuple[dict, dict]:
    """Decode header and payload from a compact JWT."""
    parts = token.split(".")
    header = json.loads(_b64url_decode(parts[0]))
    payload = json.loads(_b64url_decode(parts[1]))
    return header, payload


class TestDPoPProofGenerator:
    """Tests for DPoPProofGenerator."""

    def test_proof_is_three_part_jwt(self):
        gen = DPoPProofGenerator()
        proof = gen.generate_proof("GET", "https://example.com")
        assert re.match(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$", proof)

    def test_header_typ_and_alg(self):
        gen = DPoPProofGenerator()
        proof = gen.generate_proof("POST", "https://example.com/api")
        header, _ = _decode_jwt_parts(proof)
        assert header["typ"] == "dpop+jwt"
        assert header["alg"] == "ES256"

    def test_header_contains_jwk(self):
        gen = DPoPProofGenerator()
        proof = gen.generate_proof("GET", "https://example.com")
        header, _ = _decode_jwt_parts(proof)
        jwk = header["jwk"]
        assert jwk["kty"] == "EC"
        assert jwk["crv"] == "P-256"
        assert "x" in jwk
        assert "y" in jwk

    def test_payload_required_fields(self):
        gen = DPoPProofGenerator()
        proof = gen.generate_proof("GET", "https://example.com/resource")
        _, payload = _decode_jwt_parts(proof)
        assert payload["htm"] == "GET"
        assert payload["htu"] == "https://example.com/resource"
        assert "jti" in payload
        assert "iat" in payload

    def test_iat_is_recent(self):
        gen = DPoPProofGenerator()
        now = int(time.time())
        proof = gen.generate_proof("GET", "https://example.com")
        _, payload = _decode_jwt_parts(proof)
        assert abs(payload["iat"] - now) <= 2

    def test_jti_is_unique(self):
        gen = DPoPProofGenerator()
        jtis = set()
        for _ in range(100):
            proof = gen.generate_proof("GET", "https://example.com")
            _, payload = _decode_jwt_parts(proof)
            jtis.add(payload["jti"])
        assert len(jtis) == 100

    def test_ath_present_when_access_token_given(self):
        gen = DPoPProofGenerator()
        proof = gen.generate_proof("GET", "https://example.com", access_token="tok123")
        _, payload = _decode_jwt_parts(proof)
        assert "ath" in payload

    def test_ath_absent_when_no_access_token(self):
        gen = DPoPProofGenerator()
        proof = gen.generate_proof("GET", "https://example.com")
        _, payload = _decode_jwt_parts(proof)
        assert "ath" not in payload

    def test_method_uppercased(self):
        gen = DPoPProofGenerator()
        proof = gen.generate_proof("get", "https://example.com")
        _, payload = _decode_jwt_parts(proof)
        assert payload["htm"] == "GET"

    def test_public_jwk_matches_header(self):
        gen = DPoPProofGenerator()
        proof = gen.generate_proof("GET", "https://example.com")
        header, _ = _decode_jwt_parts(proof)
        assert header["jwk"] == gen.public_jwk

    def test_different_generators_different_keys(self):
        gen1 = DPoPProofGenerator()
        gen2 = DPoPProofGenerator()
        assert gen1.public_jwk != gen2.public_jwk
