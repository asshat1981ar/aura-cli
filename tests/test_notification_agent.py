"""Unit tests for agents/notification_agent.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.notification_agent import NotificationAgentAdapter


class TestNotificationAgentAdapterInit:
    def test_default_init(self):
        agent = NotificationAgentAdapter()
        assert agent.name == "notification"
        assert agent._mcp_url == "http://127.0.0.1:8015"
        assert agent._history == []

    def test_custom_mcp_url(self):
        agent = NotificationAgentAdapter(notification_mcp_url="http://localhost:9000")
        assert agent._mcp_url == "http://localhost:9000"


class TestRunValidation:
    def setup_method(self):
        self.agent = NotificationAgentAdapter()

    def test_missing_channel_returns_error(self):
        result = self.agent.run({"message": "hello"})
        assert "error" in result
        assert "channel" in result["error"]

    def test_missing_message_returns_error(self):
        result = self.agent.run({"channel": "slack"})
        assert "error" in result
        assert "message" in result["error"]

    @patch.object(NotificationAgentAdapter, "_send_via_mcp")
    def test_valid_call_records_history(self, mock_send):
        mock_send.return_value = {"status": "sent", "channel": "slack"}
        self.agent.run({"channel": "slack", "message": "test msg"})
        assert len(self.agent._history) == 1
        assert self.agent._history[0]["channel"] == "slack"

    @patch.object(NotificationAgentAdapter, "_send_via_mcp")
    def test_history_trimmed_at_max(self, mock_send):
        mock_send.return_value = {"status": "sent", "channel": "slack"}
        self.agent._max_history = 3
        for _ in range(5):
            self.agent.run({"channel": "slack", "message": "msg"})
        assert len(self.agent._history) == 3


class TestSendViaMcp:
    def setup_method(self):
        self.agent = NotificationAgentAdapter()

    def test_unknown_channel_returns_error(self):
        result = self.agent._send_via_mcp("fax", "hello", "info", {})
        assert result["status"] == "failed"
        assert "Unknown channel" in result["error"]

    @patch("agents.notification_agent.requests")
    def test_successful_mcp_call(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": "ok"}
        mock_requests.post.return_value = mock_resp
        result = self.agent._send_via_mcp("slack", "hello", "info", {})
        assert result["status"] == "sent"
        assert result["channel"] == "slack"

    @patch("agents.notification_agent.requests")
    def test_mcp_server_non_200_returns_failed(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_requests.post.return_value = mock_resp
        result = self.agent._send_via_mcp("slack", "hello", "info", {})
        assert result["status"] == "failed"
        assert "500" in result["error"]

    @patch("agents.notification_agent.requests")
    def test_mcp_server_unreachable_returns_failed(self, mock_requests):
        mock_requests.post.side_effect = Exception("connection refused")
        result = self.agent._send_via_mcp("slack", "hello", "info", {})
        assert result["status"] == "failed"
        assert "unreachable" in result["error"]


class TestBuildArgs:
    def setup_method(self):
        self.agent = NotificationAgentAdapter()

    def test_slack_args(self):
        args = self.agent._build_args("slack", "hello", "warning", {"channel": "#alerts"})
        assert args["channel"] == "#alerts"
        assert "[WARNING]" in args["text"]

    def test_slack_default_channel(self):
        args = self.agent._build_args("slack", "hi", "info", {})
        assert args["channel"] == "#general"

    def test_discord_args(self):
        args = self.agent._build_args("discord", "hi", "critical", {"webhook_url": "http://x"})
        assert args["webhook_url"] == "http://x"
        assert "[CRITICAL]" in args["content"]

    def test_pagerduty_args(self):
        args = self.agent._build_args("pagerduty", "down", "critical", {"routing_key": "abc"})
        assert args["routing_key"] == "abc"
        assert args["severity"] == "critical"

    def test_email_args(self):
        args = self.agent._build_args("email", "body", "info", {"to": "a@b.com", "subject": "Hi"})
        assert args["to"] == "a@b.com"
        assert args["subject"] == "Hi"
        assert args["body"] == "body"

    def test_webhook_args(self):
        args = self.agent._build_args("webhook", "msg", "info", {"url": "http://hook", "method": "PUT"})
        assert args["url"] == "http://hook"
        assert args["method"] == "PUT"
        assert args["payload"]["message"] == "msg"

    def test_unknown_channel_returns_message_only(self):
        args = self.agent._build_args("fax", "hello", "info", {})
        assert args == {"message": "hello"}


class TestGetHistory:
    def test_get_history_empty(self):
        agent = NotificationAgentAdapter()
        assert agent.get_history() == []

    @patch.object(NotificationAgentAdapter, "_send_via_mcp")
    def test_get_history_limit(self, mock_send):
        mock_send.return_value = {"status": "sent"}
        agent = NotificationAgentAdapter()
        for i in range(10):
            agent.run({"channel": "slack", "message": f"msg {i}"})
        assert len(agent.get_history(limit=5)) == 5
