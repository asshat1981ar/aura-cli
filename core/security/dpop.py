"""RFC 9449 DPoP (Demonstrating Proof-of-Possession) proof generator.

Generates DPoP proofs using ephemeral ECDSA P-256 key pairs.
Each DPoPProofGenerator instance holds one ephemeral key pair;
each call to generate_proof produces a fresh JWT with a unique jti.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
import uuid

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils


def _b64url_encode(data: bytes) -> str:
    """Base64url encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    """Base64url decode with padding restoration."""
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def _int_to_b64url(n: int, length: int) -> str:
    """Encode an integer as a base64url string of the given byte length."""
    return _b64url_encode(n.to_bytes(length, byteorder="big"))


class DPoPProofGenerator:
    """Generates DPoP proofs per RFC 9449.

    Each instance generates an ephemeral ECDSA P-256 key pair.
    The public key is embedded in the JWT header as a JWK.
    """

    def __init__(self) -> None:
        self._private_key = ec.generate_private_key(ec.SECP256R1())
        self._public_key = self._private_key.public_key()
        self._jwk = self._build_jwk()

    def _build_jwk(self) -> dict:
        """Build the JWK representation of the public key."""
        public_numbers = self._public_key.public_numbers()
        return {
            "kty": "EC",
            "crv": "P-256",
            "x": _int_to_b64url(public_numbers.x, 32),
            "y": _int_to_b64url(public_numbers.y, 32),
        }

    @property
    def public_jwk(self) -> dict:
        """Return the public JWK."""
        return dict(self._jwk)

    def generate_proof(
        self,
        htm: str,
        htu: str,
        access_token: str | None = None,
    ) -> str:
        """Generate a DPoP proof JWT.

        Args:
            htm: HTTP method (will be uppercased).
            htu: HTTP target URI.
            access_token: Optional access token for ath claim.

        Returns:
            A compact-serialized JWT string (header.payload.signature).
        """
        header = {
            "typ": "dpop+jwt",
            "alg": "ES256",
            "jwk": self._jwk,
        }

        payload: dict = {
            "jti": str(uuid.uuid4()),
            "htm": htm.upper(),
            "htu": htu,
            "iat": int(time.time()),
        }

        if access_token is not None:
            token_hash = hashlib.sha256(access_token.encode("utf-8")).digest()
            payload["ath"] = _b64url_encode(token_hash)

        header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
        payload_b64 = _b64url_encode(
            json.dumps(payload, separators=(",", ":")).encode()
        )

        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")

        # Sign with ECDSA using SHA-256
        der_signature = self._private_key.sign(
            signing_input, ec.ECDSA(hashes.SHA256())
        )

        # Convert DER signature to raw r||s format (64 bytes for P-256)
        r, s = utils.decode_dss_signature(der_signature)
        raw_signature = r.to_bytes(32, byteorder="big") + s.to_bytes(
            32, byteorder="big"
        )
        signature_b64 = _b64url_encode(raw_signature)

        return f"{header_b64}.{payload_b64}.{signature_b64}"
