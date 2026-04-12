"""Secure HTTP client with DPoP support for AURA CLI."""

from __future__ import annotations

from typing import Any, Optional

import requests

from core.security.dpop import DPoPProofGenerator, get_dpop_generator


class SecureHTTPClient:
    """HTTP client that automatically attaches DPoP proofs to requests.

    Wraps the requests library, adding:
    - DPoP proof headers on each request
    - Configurable base URL
    - Bearer token management
    """

    def __init__(
        self,
        base_url: str = "",
        access_token: Optional[str] = None,
        dpop_generator: Optional[DPoPProofGenerator] = None,
        timeout: int = 30,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._access_token = access_token
        self._dpop = dpop_generator or get_dpop_generator()
        self._timeout = timeout
        self._session = requests.Session()

    def _build_url(self, path: str) -> str:
        if path.startswith(("http://", "https://")):
            return path
        return f"{self._base_url}/{path.lstrip('/')}"

    def _prepare_headers(
        self,
        method: str,
        url: str,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> dict[str, str]:
        headers: dict[str, str] = {}
        if extra_headers:
            headers.update(extra_headers)

        # Add DPoP proof
        proof = self._dpop.generate_proof(method, url, self._access_token)
        headers["DPoP"] = proof

        # Add authorization if we have a token
        if self._access_token:
            headers["Authorization"] = f"DPoP {self._access_token}"

        return headers

    def request(
        self,
        method: str,
        path: str,
        headers: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> requests.Response:
        """Send an HTTP request with DPoP proof."""
        url = self._build_url(path)
        prepared_headers = self._prepare_headers(method.upper(), url, headers)
        kwargs.setdefault("timeout", self._timeout)
        return self._session.request(
            method, url, headers=prepared_headers, **kwargs
        )

    def get(self, path: str, **kwargs: Any) -> requests.Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> requests.Response:
        return self.request("POST", path, **kwargs)

    def put(self, path: str, **kwargs: Any) -> requests.Response:
        return self.request("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> requests.Response:
        return self.request("DELETE", path, **kwargs)
