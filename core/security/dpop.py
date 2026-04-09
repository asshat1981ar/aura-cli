"""
RFC 9449 DPoP (Demonstrating Proof of Possession) implementation.

Generates an ephemeral ECDSA P-256 key pair at process startup.
Attaches a per-request signed proof JWT to every outgoing API call,
preventing token replay attacks even if the bearer token is compromised.
"""
import base64
import hashlib
import json
import secrets
import time
from typing import Optional

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature


class DPoPProofGenerator:
    """
    Generates DPoP proof JWTs per RFC 9449 section 4.2.

    Key pair is ephemeral -- generated once per process invocation,
    stored in memory only, never persisted to disk.
    """

    def __init__(self):
        # Generate ephemeral ECDSA P-256 key pair (in memory only)
        self._private_key = ec.generate_private_key(
            ec.SECP256R1(), default_backend()
        )
        self._public_key = self._private_key.public_key()
        self._jwk = self._build_public_jwk()

    def _build_public_jwk(self) -> dict:
        """Build JWK representation of the public key per RFC 7517."""
        pub = self._public_key.public_numbers()

        def _to_base64url(n: int, length: int = 32) -> str:
            return (
                base64.urlsafe_b64encode(n.to_bytes(length, "big"))
                .rstrip(b"=")
                .decode()
            )

        return {
            "kty": "EC",
            "crv": "P-256",
            "x": _to_base64url(pub.x),
            "y": _to_base64url(pub.y),
        }

    @property
    def public_jwk(self) -> dict:
        """Return a copy of the public JWK."""
        return dict(self._jwk)

    def generate_proof(
        self,
        htm: str,
        htu: str,
        access_token: Optional[str] = None,
    ) -> str:
        """
        Generate a DPoP proof JWT for a specific request.

        Args:
            htm: HTTP method (e.g. "POST").
            htu: HTTP URI (without query/fragment).
            access_token: Bearer token for ath binding (optional).

        Returns:
            Compact-serialized JWT string for the DPoP header.
        """
        header = {
            "typ": "dpop+jwt",
            "alg": "ES256",
            "jwk": self._jwk,
        }

        payload = {
            "jti": secrets.token_urlsafe(16),
            "htm": htm.upper(),
            "htu": htu,
            "iat": int(time.time()),
        }

        # Bind to access token if provided (RFC 9449 section 4.2, ath claim)
        if access_token:
            ath = (
                base64.urlsafe_b64encode(
                    hashlib.sha256(access_token.encode()).digest()
                )
                .rstrip(b"=")
                .decode()
            )
            payload["ath"] = ath

        def _b64url(data: bytes) -> str:
            return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

        header_b64 = _b64url(json.dumps(header, separators=(",", ":")).encode())
        payload_b64 = _b64url(json.dumps(payload, separators=(",", ":")).encode())
        signing_input = f"{header_b64}.{payload_b64}".encode()

        der_sig = self._private_key.sign(signing_input, ec.ECDSA(hashes.SHA256()))

        # Convert DER signature to raw R || S (64 bytes for P-256)
        r, s = decode_dss_signature(der_sig)
        raw_sig = r.to_bytes(32, "big") + s.to_bytes(32, "big")
        sig_b64 = _b64url(raw_sig)

        return f"{header_b64}.{payload_b64}.{sig_b64}"


# Process-scoped singleton -- one key pair per CLI invocation
_dpop_generator: Optional[DPoPProofGenerator] = None


def get_dpop_generator() -> DPoPProofGenerator:
    """Return the process-scoped DPoP generator, creating it on first call."""
    global _dpop_generator
    if _dpop_generator is None:
        _dpop_generator = DPoPProofGenerator()
    return _dpop_generator


def reset_dpop_generator() -> None:
    """Reset the singleton (for testing only)."""
    global _dpop_generator
    _dpop_generator = None
