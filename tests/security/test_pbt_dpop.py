"""Property-based tests for DPoP proof generator.

Uses Hypothesis to verify structural and cryptographic properties of
DPoP proofs per RFC 9449.

Inspired by Vikram et al. (2024) "Can LLMs Write Good Property-Based Tests?"
and flyingmutant/rapid.
"""

from __future__ import annotations

import hashlib
import json
import re
import time

import pytest
from cryptography.hazmat.primitives.asymmetric import ec, utils
from cryptography.hazmat.primitives import hashes
from hypothesis import given, settings
from hypothesis import strategies as st

from core.security.dpop import DPoPProofGenerator, _b64url_decode, _b64url_encode

# Strategies
_http_methods = st.sampled_from(["GET", "POST", "PUT", "DELETE", "PATCH", "get", "post", "put", "delete", "patch"])
_urls = st.from_regex(r"https://[a-z]{1,20}\.[a-z]{2,5}(/[a-z0-9]{1,15}){0,4}", fullmatch=True)
_access_tokens = st.text(
    min_size=1,
    max_size=256,
    alphabet=st.characters(blacklist_categories=("Cs",)),
)


def _decode_jwt(token: str) -> tuple[dict, dict, bytes]:
    """Decode a compact JWT into (header, payload, signature_bytes)."""
    parts = token.split(".")
    header = json.loads(_b64url_decode(parts[0]))
    payload = json.loads(_b64url_decode(parts[1]))
    sig = _b64url_decode(parts[2])
    return header, payload, sig


# ---------------------------------------------------------------------------
# Property 1: Proof is always a valid 3-part JWT
# ---------------------------------------------------------------------------
class TestJWTStructure:
    @given(htm=_http_methods, htu=_urls)
    def test_proof_is_three_part_jwt(self, htm, htu):
        gen = DPoPProofGenerator()
        proof = gen.generate_proof(htm, htu)
        assert re.match(
            r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$", proof
        )


# ---------------------------------------------------------------------------
# Property 2: Header always contains required fields
# ---------------------------------------------------------------------------
class TestHeaderFields:
    @given(htm=_http_methods, htu=_urls)
    def test_header_has_required_fields(self, htm, htu):
        gen = DPoPProofGenerator()
        proof = gen.generate_proof(htm, htu)
        header, _, _ = _decode_jwt(proof)

        assert header["typ"] == "dpop+jwt"
        assert header["alg"] == "ES256"
        jwk = header["jwk"]
        assert jwk["kty"] == "EC"
        assert jwk["crv"] == "P-256"
        assert "x" in jwk
        assert "y" in jwk


# ---------------------------------------------------------------------------
# Property 3: Payload always contains required fields
# ---------------------------------------------------------------------------
class TestPayloadFields:
    @given(htm=_http_methods, htu=_urls)
    def test_payload_has_required_fields(self, htm, htu):
        gen = DPoPProofGenerator()
        now = int(time.time())
        proof = gen.generate_proof(htm, htu)
        _, payload, _ = _decode_jwt(proof)

        assert payload["jti"]  # non-empty
        assert payload["htm"] == htm.upper()
        assert payload["htu"] == htu
        assert isinstance(payload["iat"], int)
        assert abs(payload["iat"] - now) <= 5


# ---------------------------------------------------------------------------
# Property 4: jti uniqueness across many calls
# ---------------------------------------------------------------------------
class TestJtiUniqueness:
    @settings(max_examples=1)
    @given(st.just(None))
    def test_jti_uniqueness_at_scale(self, _):
        gen = DPoPProofGenerator()
        jtis = set()
        for _ in range(1000):
            proof = gen.generate_proof("GET", "https://example.com")
            _, payload, _ = _decode_jwt(proof)
            jtis.add(payload["jti"])
        assert len(jtis) == 1000


# ---------------------------------------------------------------------------
# Property 5: ath claim correctness
# ---------------------------------------------------------------------------
class TestAthClaim:
    @given(access_token=_access_tokens)
    def test_ath_is_correct_hash(self, access_token):
        gen = DPoPProofGenerator()
        proof = gen.generate_proof("GET", "https://example.com", access_token=access_token)
        _, payload, _ = _decode_jwt(proof)

        assert "ath" in payload
        expected = _b64url_encode(
            hashlib.sha256(access_token.encode("utf-8")).digest()
        )
        assert payload["ath"] == expected

    @given(htm=_http_methods, htu=_urls)
    def test_ath_absent_without_token(self, htm, htu):
        gen = DPoPProofGenerator()
        proof = gen.generate_proof(htm, htu)
        _, payload, _ = _decode_jwt(proof)
        assert "ath" not in payload


# ---------------------------------------------------------------------------
# Property 6: Method normalization
# ---------------------------------------------------------------------------
class TestMethodNormalization:
    @given(
        method=st.sampled_from(["get", "Get", "GET", "post", "Post", "POST"]),
        htu=_urls,
    )
    def test_htm_always_uppercased(self, method, htu):
        gen = DPoPProofGenerator()
        proof = gen.generate_proof(method, htu)
        _, payload, _ = _decode_jwt(proof)
        assert payload["htm"] == method.upper()
        assert payload["htm"] == payload["htm"].upper()


# ---------------------------------------------------------------------------
# Property 7: Signature verifies with the embedded public key
# ---------------------------------------------------------------------------
class TestSignatureVerification:
    @given(htm=_http_methods, htu=_urls)
    def test_signature_verifies(self, htm, htu):
        gen = DPoPProofGenerator()
        proof = gen.generate_proof(htm, htu)
        parts = proof.split(".")
        header, _, sig_bytes = _decode_jwt(proof)

        signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")

        # Reconstruct public key from JWK
        jwk = header["jwk"]
        x = int.from_bytes(_b64url_decode(jwk["x"]), "big")
        y = int.from_bytes(_b64url_decode(jwk["y"]), "big")
        pub_numbers = ec.EllipticCurvePublicNumbers(x, y, ec.SECP256R1())
        pub_key = pub_numbers.public_key()

        # Convert raw r||s to DER
        r = int.from_bytes(sig_bytes[:32], "big")
        s = int.from_bytes(sig_bytes[32:], "big")
        der_sig = utils.encode_dss_signature(r, s)

        # Verify — raises InvalidSignature if wrong
        pub_key.verify(der_sig, signing_input, ec.ECDSA(hashes.SHA256()))


# ---------------------------------------------------------------------------
# Property 8: Different generators produce different key material
# ---------------------------------------------------------------------------
class TestKeyUniqueness:
    @settings(max_examples=50)
    @given(st.just(None))
    def test_different_generators_different_keys(self, _):
        gen1 = DPoPProofGenerator()
        gen2 = DPoPProofGenerator()
        assert gen1.public_jwk != gen2.public_jwk
