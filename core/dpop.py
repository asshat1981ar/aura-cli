"""
DPoP (Demonstrating Proof-of-Possession) Implementation

RFC 9449 compliant DPoP support for binding access tokens to HTTP requests.
Provides token binding, proof generation, and validation capabilities.

Security Issue #427: Add DPoP Support (RFC 9449)
"""

import time
import uuid
import hashlib
import base64
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# JWT handling
try:
    import jwt

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    jwt = None

# Cryptography for key generation
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, ec, ed25519

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from core.logging_utils import log_json


class DPoPError(Exception):
    """Raised when DPoP operations fail."""

    pass


class DPoPAlgorithm(str, Enum):
    """Supported DPoP algorithms."""

    ES256 = "ES256"  # ECDSA using P-256 and SHA-256
    ES384 = "ES384"  # ECDSA using P-384 and SHA-384
    RS256 = "RS256"  # RSASSA-PKCS1-v1_5 using SHA-256
    EdDSA = "EdDSA"  # Ed25519


@dataclass
class DPoPProof:
    """Represents a DPoP proof with its associated metadata."""

    token: str  # The JWT proof token
    jkt: str  # JWK Thumbprint for key binding
    public_key: Any  # The public key used


@dataclass
class DPoPBoundToken:
    """Represents a DPoP-bound access token."""

    access_token: str
    token_type: str
    expires_in: int
    jkt: str  # JWK Thumbprint of the DPoP key


class DPoPKeyManager:
    """
    Manages DPoP key pairs for proof generation.

    Generates and stores ephemeral key pairs used to create DPoP proofs.
    Keys are stored in memory by default but can be persisted securely.
    """

    def __init__(self, algorithm: DPoPAlgorithm = DPoPAlgorithm.ES256):
        """
        Initialize the DPoP key manager.

        Args:
            algorithm: The signing algorithm to use
        """
        if not CRYPTO_AVAILABLE:
            raise DPoPError("cryptography library required for DPoP")

        self.algorithm = algorithm
        self._keys: Dict[str, Any] = {}  # kid -> private key
        self._public_keys: Dict[str, Any] = {}  # kid -> public key

    def generate_key(self, key_id: Optional[str] = None) -> str:
        """
        Generate a new DPoP key pair.

        Args:
            key_id: Optional key identifier (generated if not provided)

        Returns:
            The key identifier (kid)
        """
        kid = key_id or str(uuid.uuid4())

        if self.algorithm in (DPoPAlgorithm.ES256, DPoPAlgorithm.ES384):
            # ECDSA key
            curve = ec.SECP256R1() if self.algorithm == DPoPAlgorithm.ES256 else ec.SECP384R1()
            private_key = ec.generate_private_key(curve)
        elif self.algorithm == DPoPAlgorithm.RS256:
            # RSA key
            private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        elif self.algorithm == DPoPAlgorithm.EdDSA:
            # Ed25519 key
            private_key = ed25519.Ed25519PrivateKey.generate()
        else:
            raise DPoPError(f"Unsupported algorithm: {self.algorithm}")

        self._keys[kid] = private_key
        self._public_keys[kid] = private_key.public_key()

        log_json("INFO", "dpop_key_generated", details={"kid": kid, "alg": self.algorithm})
        return kid

    def get_private_key(self, kid: str) -> Any:
        """Get a private key by its identifier."""
        if kid not in self._keys:
            raise DPoPError(f"Key not found: {kid}")
        return self._keys[kid]

    def get_public_key(self, kid: str) -> Any:
        """Get a public key by its identifier."""
        if kid not in self._public_keys:
            raise DPoPError(f"Key not found: {kid}")
        return self._public_keys[kid]

    def get_jwk(self, kid: str) -> Dict[str, Any]:
        """
        Get the public key as a JWK (JSON Web Key).

        Args:
            kid: The key identifier

        Returns:
            JWK dictionary
        """
        public_key = self.get_public_key(kid)

        if isinstance(public_key, ec.EllipticCurvePublicKey):
            # EC key
            public_numbers = public_key.public_numbers()
            x = base64.urlsafe_b64encode(public_numbers.x.to_bytes((public_numbers.x.bit_length() + 7) // 8, "big")).decode().rstrip("=")
            y = base64.urlsafe_b64encode(public_numbers.y.to_bytes((public_numbers.y.bit_length() + 7) // 8, "big")).decode().rstrip("=")

            crv = "P-256" if self.algorithm == DPoPAlgorithm.ES256 else "P-384"

            return {
                "kty": "EC",
                "crv": crv,
                "x": x,
                "y": y,
            }

        elif isinstance(public_key, rsa.RSAPublicKey):
            # RSA key
            public_numbers = public_key.public_numbers()
            n = base64.urlsafe_b64encode(public_numbers.n.to_bytes((public_numbers.n.bit_length() + 7) // 8, "big")).decode().rstrip("=")
            e = base64.urlsafe_b64encode(public_numbers.e.to_bytes((public_numbers.e.bit_length() + 7) // 8, "big")).decode().rstrip("=")

            return {
                "kty": "RSA",
                "n": n,
                "e": e,
            }

        elif isinstance(public_key, ed25519.Ed25519PublicKey):
            # Ed25519 key
            raw_bytes = public_key.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
            x = base64.urlsafe_b64encode(raw_bytes).decode().rstrip("=")

            return {
                "kty": "OKP",
                "crv": "Ed25519",
                "x": x,
            }

        raise DPoPError(f"Unsupported key type: {type(public_key)}")

    def compute_jkt(self, kid: str) -> str:
        """
        Compute the JWK Thumbprint (jkt) for a key.

        Args:
            kid: The key identifier

        Returns:
            Base64URL-encoded JWK Thumbprint
        """
        jwk = self.get_jwk(kid)
        # Required members only for thumbprint
        thumbprint_jwk = {k: v for k, v in jwk.items() if k in ["kty", "crv", "x", "y", "n", "e"]}

        # Canonical JSON encoding
        import json

        jwk_json = json.dumps(thumbprint_jwk, separators=(",", ":"), sort_keys=True)

        # SHA-256 hash
        digest = hashlib.sha256(jwk_json.encode()).digest()
        return base64.urlsafe_b64encode(digest).decode().rstrip("=")

    def delete_key(self, kid: str) -> bool:
        """Delete a key pair."""
        if kid in self._keys:
            del self._keys[kid]
            del self._public_keys[kid]
            return True
        return False


class DPoPProofGenerator:
    """
    Generates DPoP proofs for HTTP requests.

    Implements RFC 9449 DPoP proof JWT generation with proper claims.
    """

    def __init__(self, key_manager: Optional[DPoPKeyManager] = None):
        """
        Initialize the proof generator.

        Args:
            key_manager: Optional key manager (creates default if not provided)
        """
        if not JWT_AVAILABLE:
            raise DPoPError("PyJWT library required for DPoP")

        self.key_manager = key_manager or DPoPKeyManager()

    def generate_proof(
        self,
        htm: str,  # HTTP method
        htu: str,  # HTTP target URI
        kid: Optional[str] = None,
        ath: Optional[str] = None,  # Access token hash (for sender-constrained tokens)
        nonce: Optional[str] = None,  # Server-provided nonce
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> DPoPProof:
        """
        Generate a DPoP proof JWT.

        Args:
            htm: HTTP method (GET, POST, etc.)
            htu: HTTP target URI
            kid: Key identifier (generates new key if not provided)
            ath: Access token hash for sender-constrained tokens
            nonce: Server-provided nonce
            additional_claims: Additional claims to include

        Returns:
            DPoPProof containing the JWT and metadata
        """
        # Generate key if needed
        if kid is None:
            kid = self.key_manager.generate_key()

        # Get JWK
        jwk = self.key_manager.get_jwk(kid)
        jwk["kid"] = kid

        # Compute JKT for binding
        jkt = self.key_manager.compute_jkt(kid)

        # Build claims
        now = int(time.time())
        claims = {
            # Required claims per RFC 9449
            "jti": str(uuid.uuid4()),  # Unique token ID
            "htm": htm.upper(),  # HTTP method
            "htu": htu,  # HTTP URI
            "iat": now,  # Issued at
            # Public key confirmation
            "jwk": jwk,
        }

        # Optional claims
        if ath:
            claims["ath"] = ath
        if nonce:
            claims["nonce"] = nonce

        # Add any additional claims
        if additional_claims:
            claims.update(additional_claims)

        # Create JWT header
        header = {
            "typ": "dpop+jwt",
            "alg": self.key_manager.algorithm,
            "jwk": jwk,
        }

        # Sign the JWT
        private_key = self.key_manager.get_private_key(kid)
        token = jwt.encode(
            claims,
            key=private_key,
            algorithm=self.key_manager.algorithm,
            headers=header,
        )

        log_json("DEBUG", "dpop_proof_generated", details={"htm": htm, "htu": htu, "kid": kid})

        return DPoPProof(token=token, jkt=jkt, public_key=self.key_manager.get_public_key(kid))

    def compute_ath(self, access_token: str) -> str:
        """
        Compute the access token hash (ath) claim.

        Args:
            access_token: The access token

        Returns:
            Base64URL-encoded SHA-256 hash of the access token
        """
        digest = hashlib.sha256(access_token.encode()).digest()
        return base64.urlsafe_b64encode(digest).decode().rstrip("=")


class DPoPValidator:
    """
    Validates DPoP proofs for incoming requests.

    Implements RFC 9449 validation requirements.
    """

    def __init__(self, max_iat_offset: int = 300):
        """
        Initialize the validator.

        Args:
            max_iat_offset: Maximum allowed time offset for iat claim (seconds)
        """
        if not JWT_AVAILABLE:
            raise DPoPError("PyJWT library required for DPoP validation")

        self.max_iat_offset = max_iat_offset
        self._used_jtis: set = set()  # Track used jti values for replay protection

    def validate(
        self,
        proof_token: str,
        htm: str,
        htu: str,
        access_token: Optional[str] = None,
        nonce: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate a DPoP proof.

        Args:
            proof_token: The DPoP proof JWT
            htm: Expected HTTP method
            htu: Expected HTTP URI
            access_token: Optional access token for ath validation
            nonce: Expected nonce (if server requires it)

        Returns:
            Dictionary with validation result and claims

        Raises:
            DPoPError: If validation fails
        """
        try:
            # Decode without verification first to get header
            jwt.decode(proof_token, options={"verify_signature": False})
            header = jwt.get_unverified_header(proof_token)
        except jwt.InvalidTokenError as e:
            raise DPoPError(f"Invalid JWT format: {e}")

        # Validate header
        if header.get("typ") != "dpop+jwt":
            raise DPoPError(f"Invalid typ header: {header.get('typ')}")

        if header.get("alg") not in ["ES256", "ES384", "RS256", "EdDSA"]:
            raise DPoPError(f"Invalid algorithm: {header.get('alg')}")

        # Get JWK from header
        jwk = header.get("jwk")
        if not jwk:
            raise DPoPError("Missing jwk in header")

        if jwk.get("kty") not in ["EC", "RSA", "OKP"]:
            raise DPoPError(f"Invalid key type: {jwk.get('kty')}")

        # Reconstruct public key for verification
        public_key = self._jwk_to_public_key(jwk)

        # Verify signature
        try:
            claims = jwt.decode(
                proof_token,
                key=public_key,
                algorithms=[header["alg"]],
            )
        except jwt.InvalidSignatureError:
            raise DPoPError("Invalid signature")
        except jwt.ExpiredSignatureError:
            raise DPoPError("Token expired")
        except jwt.InvalidTokenError as e:
            raise DPoPError(f"Token validation failed: {e}")

        # Validate required claims
        if "jti" not in claims:
            raise DPoPError("Missing jti claim")

        if "htm" not in claims or "htu" not in claims:
            raise DPoPError("Missing htm or htu claim")

        if "iat" not in claims:
            raise DPoPError("Missing iat claim")

        # Check for replay
        if claims["jti"] in self._used_jtis:
            raise DPoPError("jti replay detected")
        self._used_jtis.add(claims["jti"])

        # Validate iat timestamp
        now = int(time.time())
        iat = claims["iat"]
        if abs(now - iat) > self.max_iat_offset:
            raise DPoPError(f"iat timestamp too far from current time: {abs(now - iat)}s")

        # Validate HTTP method
        if claims["htm"].upper() != htm.upper():
            raise DPoPError(f"htm mismatch: expected {htm}, got {claims['htm']}")

        # Validate HTTP URI (normalize for comparison)
        normalized_htu = self._normalize_uri(claims["htu"])
        expected_htu = self._normalize_uri(htu)
        if normalized_htu != expected_htu:
            raise DPoPError(f"htu mismatch: expected {htu}, got {claims['htu']}")

        # Validate access token hash if provided
        if access_token and "ath" in claims:
            expected_ath = self._compute_ath(access_token)
            if claims["ath"] != expected_ath:
                raise DPoPError("ath mismatch")

        # Validate nonce if required
        if nonce and claims.get("nonce") != nonce:
            raise DPoPError("nonce mismatch")

        # Compute JWK Thumbprint
        jkt = self._compute_jkt(jwk)

        log_json("DEBUG", "dpop_proof_validated", details={"htm": htm, "htu": htu, "jkt": jkt[:16] + "..."})

        return {
            "valid": True,
            "claims": claims,
            "jkt": jkt,
            "jwk": jwk,
        }

    def _jwk_to_public_key(self, jwk: Dict[str, Any]) -> Any:
        """Convert JWK to public key object."""
        if not CRYPTO_AVAILABLE:
            raise DPoPError("cryptography library required for key reconstruction")

        kty = jwk.get("kty")

        if kty == "EC":
            crv = jwk.get("crv")
            if crv == "P-256":
                curve = ec.SECP256R1()
            elif crv == "P-384":
                curve = ec.SECP384R1()
            else:
                raise DPoPError(f"Unsupported curve: {crv}")

            # Decode x and y coordinates
            x = int.from_bytes(base64.urlsafe_b64decode(jwk["x"] + "=="), "big")
            y = int.from_bytes(base64.urlsafe_b64decode(jwk["y"] + "=="), "big")

            return ec.EllipticCurvePublicNumbers(x, y, curve).public_key()

        elif kty == "RSA":
            n = int.from_bytes(base64.urlsafe_b64decode(jwk["n"] + "=="), "big")
            e = int.from_bytes(base64.urlsafe_b64decode(jwk["e"] + "=="), "big")

            return rsa.RSAPublicNumbers(e, n).public_key()

        elif kty == "OKP" and jwk.get("crv") == "Ed25519":
            raw_bytes = base64.urlsafe_b64decode(jwk["x"] + "==")
            return ed25519.Ed25519PublicKey.from_public_bytes(raw_bytes)

        raise DPoPError(f"Unsupported JWK: {kty}")

    def _compute_jkt(self, jwk: Dict[str, Any]) -> str:
        """Compute JWK Thumbprint."""
        thumbprint_jwk = {k: v for k, v in jwk.items() if k in ["kty", "crv", "x", "y", "n", "e"]}
        import json

        jwk_json = json.dumps(thumbprint_jwk, separators=(",", ":"), sort_keys=True)
        digest = hashlib.sha256(jwk_json.encode()).digest()
        return base64.urlsafe_b64encode(digest).decode().rstrip("=")

    def _compute_ath(self, access_token: str) -> str:
        """Compute access token hash."""
        digest = hashlib.sha256(access_token.encode()).digest()
        return base64.urlsafe_b64encode(digest).decode().rstrip("=")

    def _normalize_uri(self, uri: str) -> str:
        """Normalize URI for comparison."""
        # Strip fragment
        if "#" in uri:
            uri = uri.split("#")[0]

        # Normalize case for scheme and host
        if "://" in uri:
            scheme, rest = uri.split("://", 1)
            if "/" in rest:
                host, path = rest.split("/", 1)
                # Don't normalize host (case-sensitive in some cases)
                uri = f"{scheme.lower()}://{host}/{path}"
            else:
                uri = f"{scheme.lower()}://{rest}"

        return uri


class DPoPHTTPClient:
    """
    HTTP client with DPoP proof injection.

    Automatically generates and attaches DPoP proofs to HTTP requests.
    """

    def __init__(
        self,
        key_manager: Optional[DPoPKeyManager] = None,
        proof_generator: Optional[DPoPProofGenerator] = None,
    ):
        """
        Initialize the DPoP HTTP client.

        Args:
            key_manager: Optional key manager
            proof_generator: Optional proof generator
        """
        self.key_manager = key_manager or DPoPKeyManager()
        self.proof_generator = proof_generator or DPoPProofGenerator(self.key_manager)
        self._default_kid: Optional[str] = None

    def set_default_key(self, kid: str) -> None:
        """Set the default key for proof generation."""
        self._default_kid = kid

    def generate_default_key(self) -> str:
        """Generate and set a default key."""
        self._default_kid = self.key_manager.generate_key()
        return self._default_kid

    def prepare_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        access_token: Optional[str] = None,
    ) -> Tuple[Dict[str, str], DPoPProof]:
        """
        Prepare request headers with DPoP proof.

        Args:
            method: HTTP method
            url: Request URL
            headers: Existing headers (optional)
            access_token: Access token for sender-constrained tokens

        Returns:
            Tuple of (updated headers, DPoP proof)
        """
        headers = headers or {}

        # Use default key or generate one
        kid = self._default_kid
        if not kid:
            kid = self.generate_default_key()

        # Compute ath if access token provided
        ath = None
        if access_token:
            ath = self.proof_generator.compute_ath(access_token)

        # Generate proof
        proof = self.proof_generator.generate_proof(
            htm=method,
            htu=url,
            kid=kid,
            ath=ath,
        )

        # Add DPoP header
        headers["DPoP"] = proof.token

        return headers, proof


def create_dpop_bound_request(
    method: str,
    url: str,
    access_token: Optional[str] = None,
) -> Tuple[Dict[str, str], str]:
    """
    Convenience function to create DPoP-bound request headers.

    Args:
        method: HTTP method
        url: Request URL
        access_token: Optional access token

    Returns:
        Tuple of (headers, jkt) where jkt is the key thumbprint for binding
    """
    client = DPoPHTTPClient()
    headers, proof = client.prepare_request(method, url, access_token=access_token)
    return headers, proof.jkt
