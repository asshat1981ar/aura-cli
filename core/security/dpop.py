"""DPoP (Demonstrating Proof-of-Possession) proof generator.

Implements RFC 9449 — DPoP proof JWTs for sender-constrained access tokens.
Uses ephemeral EC P-256 key pairs; each ``DPoPProofGenerator`` instance holds
one key pair for its lifetime.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _int_to_b64url(n: int, length: int) -> str:
    return _b64url(n.to_bytes(length, byteorder="big"))


class DPoPProofGenerator:
    """Generate DPoP proof JWTs per RFC 9449.

    Each instance creates an ephemeral EC P-256 key pair.
    """

    def __init__(self) -> None:
        self._private_key = ec.generate_private_key(ec.SECP256R1())
        self._public_key = self._private_key.public_key()
        pub_numbers = self._public_key.public_numbers()
        self._jwk = {
            "kty": "EC",
            "crv": "P-256",
            "x": _int_to_b64url(pub_numbers.x, 32),
            "y": _int_to_b64url(pub_numbers.y, 32),
        }

    def generate_proof(
        self,
        method: str,
        url: str,
        *,
        access_token: str | None = None,
    ) -> str:
        """Generate a DPoP proof JWT.

        Parameters
        ----------
        method:
            HTTP method (GET, POST, etc.).
        url:
            The HTTP request URI.
        access_token:
            If provided, the ``ath`` (access token hash) claim is included.
        """
        header = {
            "typ": "dpop+jwt",
            "alg": "ES256",
            "jwk": self._jwk,
        }

        payload: dict[str, object] = {
            "htm": method.upper(),
            "htu": url,
            "iat": int(time.time()),
            "jti": secrets.token_urlsafe(16),
        }

        if access_token is not None:
            token_hash = hashlib.sha256(access_token.encode("ascii")).digest()
            payload["ath"] = _b64url(token_hash)

        header_b64 = _b64url(json.dumps(header, separators=(",", ":")).encode())
        payload_b64 = _b64url(json.dumps(payload, separators=(",", ":")).encode())

        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        der_sig = self._private_key.sign(signing_input, ec.ECDSA(hashes.SHA256()))

        r, s = decode_dss_signature(der_sig)
        sig_bytes = r.to_bytes(32, "big") + s.to_bytes(32, "big")
        sig_b64 = _b64url(sig_bytes)

        return f"{header_b64}.{payload_b64}.{sig_b64}"
