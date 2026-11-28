"""DPoP (Demonstrating Proof-of-Possession) proof generator.

Implements RFC 9449 DPoP proof tokens for OAuth 2.0.
Generates signed JWT proofs that bind an access token to a specific
HTTP method and URL.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
import uuid
from typing import Optional

# Use hmac-based signing as a lightweight fallback when
# cryptographic libraries (jwcrypto, PyJWT) are not available.
import hmac


def _b64url_encode(data: bytes) -> str:
    """Base64url-encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    """Base64url-decode with padding restoration."""
    padding = "=" * (4 - len(s) % 4)
    return base64.urlsafe_b64decode(s + padding)


class DPoPProofGenerator:
    """Generates DPoP proof JWTs per RFC 9449.

    Uses HMAC-SHA256 as a lightweight signing mechanism.
    In production, replace with EC or RSA key pair signing.
    """

    def __init__(self, signing_key: bytes | None = None) -> None:
        if signing_key is None:
            signing_key = uuid.uuid4().bytes  # Ephemeral key per instance
        self._signing_key = signing_key

    def generate_proof(
        self,
        htm: str,
        htu: str,
        access_token: Optional[str] = None,
    ) -> str:
        """Generate a DPoP proof JWT.

        Args:
            htm: HTTP method (e.g., "GET", "POST").
            htu: HTTP target URI.
            access_token: Optional access token to bind via ath claim.

        Returns:
            A compact JWT string (header.payload.signature).

        Raises:
            ValueError: If htm or htu are empty.
        """
        if not htm:
            raise ValueError("htm (HTTP method) must not be empty")
        if not htu:
            raise ValueError("htu (HTTP target URI) must not be empty")

        header = {
            "typ": "dpop+jwt",
            "alg": "HS256",
        }

        payload: dict = {
            "jti": str(uuid.uuid4()),
            "htm": str(htm),
            "htu": str(htu),
            "iat": int(time.time()),
        }

        if access_token is not None:
            # ath = base64url(SHA-256(access_token))
            token_bytes = str(access_token).encode("utf-8", errors="replace")
            ath = _b64url_encode(hashlib.sha256(token_bytes).digest())
            payload["ath"] = ath

        header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
        payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())

        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        signature = hmac.new(self._signing_key, signing_input, hashlib.sha256).digest()
        sig_b64 = _b64url_encode(signature)

        return f"{header_b64}.{payload_b64}.{sig_b64}"


# Module-level singleton
_dpop_generator: Optional[DPoPProofGenerator] = None


def get_dpop_generator() -> DPoPProofGenerator:
    """Get or create the module-level DPoP generator."""
    global _dpop_generator
    if _dpop_generator is None:
        _dpop_generator = DPoPProofGenerator()
    return _dpop_generator
