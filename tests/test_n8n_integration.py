from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from core.integrations.n8n import N8NEvent, N8NIntegrationConfig, emit_n8n_event, load_n8n_config


def test_load_n8n_config_defaults_disabled():
    cfg = load_n8n_config({})
    assert cfg.enabled is False
    assert cfg.webhook_url == ""


def test_emit_n8n_event_returns_disabled_when_off():
    result = emit_n8n_event(
        N8NEvent(event_type="sadd.session.started", payload={"session_id": "s1"}),
        N8NIntegrationConfig(enabled=False),
    )
    assert result["status"] == "disabled"


def test_emit_n8n_event_posts_json_payload():
    fake_response = MagicMock()
    fake_response.__enter__.return_value = fake_response
    fake_response.read.return_value = json.dumps({"ok": True}).encode("utf-8")
    fake_response.status = 200

    with patch("urllib.request.urlopen", return_value=fake_response) as mock_open:
        result = emit_n8n_event(
            N8NEvent(event_type="aura.goal.completed", payload={"goal": "test"}),
            N8NIntegrationConfig(enabled=True, webhook_url="http://localhost:5678/webhook/aura-events"),
        )

    assert result["status"] == "ok"
    request = mock_open.call_args.args[0]
    assert request.full_url == "http://localhost:5678/webhook/aura-events"


def test_load_n8n_config_supports_legacy_n8n_connector_shape():
    cfg = load_n8n_config(
        {
            "n8n_connector": {
                "enabled": True,
                "notification_webhook": "http://localhost:5678/webhook/aura-notify",
                "timeout_seconds": 3.5,
            }
        }
    )
    assert cfg.enabled is True
    assert cfg.webhook_url == "http://localhost:5678/webhook/aura-notify"
    assert cfg.timeout_seconds == 3.5
