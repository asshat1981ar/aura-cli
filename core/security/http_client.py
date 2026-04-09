"""HTTP client with DPoP proof-of-possession support."""

from __future__ import annotations

import requests

from core.security.dpop import DPoPProofGenerator


class DPoPSession(requests.Session):
    """A requests.Session that automatically attaches DPoP proof headers.

    Each session holds its own ephemeral key pair via ``DPoPProofGenerator``.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._dpop = DPoPProofGenerator()

    def request(self, method: str, url: str, **kwargs: object) -> requests.Response:
        access_token = None
        if "headers" in kwargs and isinstance(kwargs["headers"], dict):
            auth = kwargs["headers"].get("Authorization", "")
            if isinstance(auth, str) and auth.startswith("DPoP "):
                access_token = auth[len("DPoP ") :]

        proof = self._dpop.generate_proof(method, url, access_token=access_token)

        headers = dict(kwargs.pop("headers", None) or {})  # type: ignore[arg-type]
        headers["DPoP"] = proof
        kwargs["headers"] = headers

        return super().request(method, url, **kwargs)
