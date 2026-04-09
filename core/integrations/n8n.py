from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from core.security.http_client import attach_dpop_headers


@dataclass(frozen=True)
class N8NIntegrationConfig:
    enabled: bool = False
    webhook_url: str = ""
    auth_header: str = ""
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class N8NEvent:
    event_type: str
    payload: dict[str, Any]
    source: str = "aura-cli"

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "source": self.source,
            "payload": self.payload,
        }


def _nested_dict(config: dict[str, Any] | None, *keys: str) -> dict[str, Any]:
    node: Any = config or {}
    for key in keys:
        if not isinstance(node, dict):
            return {}
        node = node.get(key, {})
    return node if isinstance(node, dict) else {}


def load_n8n_config(config: dict[str, Any] | None = None) -> N8NIntegrationConfig:
    integrations_n8n = _nested_dict(config, "integrations", "n8n")
    connector_n8n = _nested_dict(config, "n8n_connector")

    webhook_default = str(
        integrations_n8n.get(
            "webhook_url",
            connector_n8n.get("notification_webhook", ""),
        )
    )
    auth_default = str(
        integrations_n8n.get(
            "auth_header",
            connector_n8n.get("auth_header", ""),
        )
    )
    enabled_default = integrations_n8n.get(
        "enabled",
        connector_n8n.get("enabled", False),
    )
    timeout_default = integrations_n8n.get(
        "timeout_seconds",
        connector_n8n.get("timeout_seconds", 10.0),
    )

    webhook_url = os.getenv("AURA_N8N_WEBHOOK_URL", webhook_default)
    auth_header = os.getenv("AURA_N8N_AUTH_HEADER", auth_default)
    enabled = str(os.getenv("AURA_N8N_ENABLED", str(enabled_default))).lower() in {"1", "true", "yes", "on"}
    timeout_raw = os.getenv("AURA_N8N_TIMEOUT_SECONDS", str(timeout_default))
    try:
        timeout_seconds = float(timeout_raw)
    except (TypeError, ValueError):
        timeout_seconds = 10.0
    return N8NIntegrationConfig(
        enabled=enabled,
        webhook_url=webhook_url,
        auth_header=auth_header,
        timeout_seconds=timeout_seconds,
    )


def emit_n8n_event(event: N8NEvent, cfg: N8NIntegrationConfig) -> dict[str, Any]:
    if not cfg.enabled:
        return {"status": "disabled"}
    if not cfg.webhook_url:
        return {"status": "misconfigured", "reason": "missing_webhook_url"}

    headers = {"Content-Type": "application/json"}
    if cfg.auth_header:
        headers["Authorization"] = cfg.auth_header
    attach_dpop_headers(headers, "POST", cfg.webhook_url)
    body = json.dumps(event.as_dict()).encode("utf-8")
    request = urllib.request.Request(cfg.webhook_url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=cfg.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            return {
                "status": "ok",
                "http_status": response.status,
                "response": json.loads(raw) if raw else {},
            }
    except urllib.error.HTTPError as exc:
        return {"status": "http_error", "http_status": exc.code, "reason": str(exc)}
    except urllib.error.URLError as exc:
        return {"status": "network_error", "reason": str(exc.reason)}
