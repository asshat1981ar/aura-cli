"""Tests for aura_cli/github_integration.py."""

import json
import hashlib
import hmac
from unittest.mock import Mock, patch, MagicMock

import pytest


class TestGitHubAppInit:
    """Test GitHubApp initialization."""

    def test_init_without_pygithub_allows_basic_ops(self):
        """Test initialization works without PyGithub for basic operations."""
        # Should be able to create instance without PyGithub
        from aura_cli.github_integration import GitHubApp

        app = GitHubApp("app_id", "private_key", "webhook_secret")
        assert app.app_id == "app_id"
        assert app.webhook_secret == "webhook_secret"


class TestVerifyWebhookSignature:
    """Test webhook signature verification."""

    def test_valid_signature(self):
        """Test valid signature verification."""
        from aura_cli.github_integration import GitHubApp

        app = GitHubApp("app_id", "private_key", "secret")

        payload = b'{"test": "data"}'
        signature = (
            "sha256="
            + hmac.new(
                b"secret",
                payload,
                hashlib.sha256,
            ).hexdigest()
        )

        assert app.verify_webhook_signature(payload, signature) is True

    def test_invalid_signature(self):
        """Test invalid signature verification."""
        from aura_cli.github_integration import GitHubApp

        app = GitHubApp("app_id", "private_key", "secret")

        payload = b'{"test": "data"}'
        signature = "sha256=invalid_signature"

        assert app.verify_webhook_signature(payload, signature) is False

    def test_no_secret_always_true(self):
        """Test verification without secret always returns True."""
        from aura_cli.github_integration import GitHubApp

        app = GitHubApp("app_id", "private_key", None)

        assert app.verify_webhook_signature(b"test", "any") is True


class TestHandleWebhook:
    """Test webhook event handling."""

    def test_handle_pull_request_opened(self):
        """Test handling PR opened event."""
        from aura_cli.github_integration import GitHubApp

        app = GitHubApp("app_id", "private_key")

        payload = {
            "action": "opened",
            "pull_request": {"number": 123, "title": "Test PR"},
            "repository": {"full_name": "owner/repo"},
        }

        result = app.handle_webhook("pull_request", payload)

        assert result["status"] == "processed"

    def test_handle_unsupported_event(self):
        """Test handling unsupported event."""
        from aura_cli.github_integration import GitHubApp

        app = GitHubApp("app_id", "private_key")

        payload = {"test": "data"}

        result = app.handle_webhook("unknown_event", payload)

        assert result["status"] == "ignored"

    def test_handle_installation(self):
        """Test handling installation event."""
        from aura_cli.github_integration import GitHubApp

        app = GitHubApp("app_id", "private_key")

        payload = {
            "action": "created",
            "installation": {
                "id": 12345,
                "account": {"login": "testuser"},
            },
        }

        result = app.handle_webhook("installation", payload)

        assert result["status"] == "processed"


class TestProcessCommand:
    """Test slash command processing."""

    def test_review_command(self):
        """Test /aura review command."""
        from aura_cli.github_integration import GitHubApp

        app = GitHubApp("app_id", "private_key")

        body = "/aura review"
        context = {"number": 123}

        result = app._process_command(body, context, "pr_comment")

        assert result["status"] == "accepted"
        assert result["command"] == "review"

    def test_fix_command(self):
        """Test /aura fix command."""
        from aura_cli.github_integration import GitHubApp

        app = GitHubApp("app_id", "private_key")

        body = "/aura fix"
        context = {"number": 123}

        result = app._process_command(body, context, "pr_comment")

        assert result["status"] == "accepted"
        assert result["command"] == "fix"

    def test_help_command(self):
        """Test /aura help command."""
        from aura_cli.github_integration import GitHubApp

        app = GitHubApp("app_id", "private_key")

        body = "/aura help"
        context = {"number": 123}

        result = app._process_command(body, context, "pr_comment")

        assert result["status"] == "success"
        assert result["command"] == "help"
        assert "Available commands" in result["message"]

    def test_no_command(self):
        """Test comment without command."""
        from aura_cli.github_integration import GitHubApp

        app = GitHubApp("app_id", "private_key")

        body = "This is just a regular comment"
        context = {"number": 123}

        result = app._process_command(body, context, "pr_comment")

        assert result["status"] == "ignored"


class TestGlobalInstance:
    """Test global GitHub App instance functions."""

    def test_get_github_app_before_init(self):
        """Test getting app before initialization returns None."""
        from aura_cli.github_integration import get_github_app

        # Reset global instance
        import aura_cli.github_integration as gh_module

        gh_module._github_app = None

        assert get_github_app() is None
