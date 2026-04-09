"""
DPoP-aware HTTP helpers.

Provides ``attach_dpop_headers`` for use at every call site that sends
a bearer token, and ``DPoPSession`` as a drop-in ``requests.Session``
wrapper that does it automatically.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests

from core.security.dpop import get_dpop_generator


def _strip_query_fragment(url: str) -> str:
    """Return the URL without query string or fragment (RFC 9449 section 4.2)."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def attach_dpop_headers(
    headers: Dict[str, str],
    method: str,
    url: str,
    access_token: Optional[str] = None,
) -> Dict[str, str]:
    """
    Add a ``DPoP`` proof header to *headers* in-place and return them.

    If *access_token* is provided the proof's ``ath`` claim is bound to it.
    """
    htu = _strip_query_fragment(url)
    dpop = get_dpop_generator()
    headers["DPoP"] = dpop.generate_proof(method, htu, access_token=access_token)
    return headers


class DPoPSession:
    """
    ``requests.Session`` wrapper that automatically attaches DPoP proof
    headers to every outgoing request that carries a bearer token.
    """

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self._session = session or requests.Session()
        self._dpop = get_dpop_generator()

    def request(
        self,
        method: str,
        url: str,
        access_token: Optional[str] = None,
        **kwargs: Any,
    ) -> requests.Response:
        headers: Dict[str, str] = dict(kwargs.pop("headers", {}) or {})
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
            htu = _strip_query_fragment(url)
            headers["DPoP"] = self._dpop.generate_proof(method, htu, access_token)
        kwargs["headers"] = headers
        return self._session.request(method, url, **kwargs)

    def get(self, url: str, **kw: Any) -> requests.Response:
        return self.request("GET", url, **kw)

    def post(self, url: str, **kw: Any) -> requests.Response:
        return self.request("POST", url, **kw)

    def put(self, url: str, **kw: Any) -> requests.Response:
        return self.request("PUT", url, **kw)

    def patch(self, url: str, **kw: Any) -> requests.Response:
        return self.request("PATCH", url, **kw)

    def delete(self, url: str, **kw: Any) -> requests.Response:
        return self.request("DELETE", url, **kw)
