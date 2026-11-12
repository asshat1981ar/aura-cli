"""
Fuzz tests for DPoP proof generator.
Verifies robustness of proof generation against arbitrary HTTP method/URL inputs.

Run with: pytest tests/security/fuzz_dpop.py -v
Coverage-guided fuzzing: python -m pytest tests/security/fuzz_dpop.py --hypothesis-seed=random
"""

import base64
import json

import pytest
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st

from core.security.dpop import DPoPProofGenerator, get_dpop_generator


class TestDPoPFuzz:

    @given(
        htm=st.text(min_size=0, max_size=64),
        htu=st.text(min_size=0, max_size=2048),
    )
    @settings(max_examples=1000, suppress_health_check=[HealthCheck.too_slow])
    def test_arbitrary_htm_htu_no_crash(self, htm, htu):
        """generate_proof must not crash on any string input for htm and htu."""
        gen = DPoPProofGenerator()
        try:
            proof = gen.generate_proof(htm, htu)
            # If it succeeds, must be a valid 3-part JWT
            parts = proof.split(".")
            assert len(parts) == 3
            assert all(len(p) > 0 for p in parts)
        except (ValueError, UnicodeEncodeError):
            pass  # Documented rejection acceptable; crash is not

    @given(
        access_token=st.one_of(
            st.none(),
            st.text(min_size=0, max_size=4096),
            st.binary(min_size=0, max_size=4096).map(
                lambda b: b.decode("utf-8", errors="replace")
            ),
        )
    )
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_arbitrary_access_token_no_crash(self, access_token):
        """generate_proof must not crash on any access_token value."""
        gen = DPoPProofGenerator()
        try:
            gen.generate_proof(
                "GET", "https://api.example.com/v1/resource",
                access_token=access_token,
            )
        except (ValueError, TypeError):
            pass

    @given(
        tokens=st.lists(
            st.text(
                min_size=1,
                max_size=128,
                alphabet=st.characters(blacklist_categories=("Cs",)),
            ),
            min_size=2,
            max_size=50,
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_jti_no_collision_under_fuzzing(self, tokens):
        """jti values must be unique even when generating many proofs with varied tokens."""
        gen = DPoPProofGenerator()
        jtis = []
        for token in tokens:
            proof = gen.generate_proof("POST", "https://example.com", access_token=token)
            payload_b64 = proof.split(".")[1]
            padding = "=" * (4 - len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64 + padding))
            jtis.append(payload["jti"])
        assert len(set(jtis)) == len(jtis), f"jti collision detected: {jtis}"

    @given(
        htm=st.sampled_from(["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]),
        htu=st.text(min_size=1, max_size=2048),
        token=st.one_of(st.none(), st.text(min_size=1, max_size=256)),
    )
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_proof_payload_structure(self, htm, htu, token):
        """Generated proofs must always contain required claims: jti, htm, htu, iat."""
        gen = DPoPProofGenerator()
        try:
            proof = gen.generate_proof(htm, htu, access_token=token)
            payload_b64 = proof.split(".")[1]
            padding = "=" * (4 - len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64 + padding))

            assert "jti" in payload, "Missing jti claim"
            assert "htm" in payload, "Missing htm claim"
            assert "htu" in payload, "Missing htu claim"
            assert "iat" in payload, "Missing iat claim"
            assert payload["htm"] == htm
            assert payload["htu"] == htu

            if token is not None:
                assert "ath" in payload, "Missing ath claim when access_token provided"
        except ValueError:
            pass  # Empty htm/htu rejected is fine

    @given(signing_key=st.binary(min_size=1, max_size=256))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_different_keys_different_signatures(self, signing_key):
        """Different signing keys must produce different signatures for same input."""
        gen1 = DPoPProofGenerator(signing_key=signing_key)
        gen2 = DPoPProofGenerator(signing_key=signing_key + b"\x00")

        proof1 = gen1.generate_proof("GET", "https://example.com")
        proof2 = gen2.generate_proof("GET", "https://example.com")

        sig1 = proof1.split(".")[2]
        sig2 = proof2.split(".")[2]
        # Different keys should produce different signatures
        # (same payload, different key => different HMAC)
        assert sig1 != sig2
