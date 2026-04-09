"""DPoP-aware HTTP session wrapper.

Automatically attaches DPoP proof headers to outgoing requests.
"""

from __future__ import annotations

from typing import Any

import requests

from core.security.dpop import DPoPProofGenerator


class DPoPSession:
    """HTTP session that automatically attaches DPoP proofs.

    Wraps requests.Session to inject a DPoP header on every request.
    """

    def __init__(
        self,
        access_token: str | None = None,
        generator: DPoPProofGenerator | None = None,
    ) -> None:
        self._session = requests.Session()
        self._generator = generator or DPoPProofGenerator()
        self._access_token = access_token

    @property
    def generator(self) -> DPoPProofGenerator:
        """Return the DPoP proof generator."""
        return self._generator

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        """Send an HTTP request with a DPoP proof header.

        Args:
            method: HTTP method (GET, POST, etc.).
            url: Target URL.
            **kwargs: Additional arguments passed to requests.Session.request.

        Returns:
            The HTTP response.
        """
        proof = self._generator.generate_proof(
            htm=method,
            htu=url,
            access_token=self._access_token,
        )
        headers = kwargs.pop("headers", {}) or {}
        headers["DPoP"] = proof
        if self._access_token:
            headers.setdefault("Authorization", f"DPoP {self._access_token}")
        return self._session.request(method, url, headers=headers, **kwargs)

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> requests.Response:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> requests.Response:
        return self.request("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> requests.Response:
        return self.request("DELETE", url, **kwargs)

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> DPoPSession:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
