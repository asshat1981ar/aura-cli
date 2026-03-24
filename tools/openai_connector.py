"""OpenAI Connector - Bridge between OpenAI Aura Dev Suite and local aura-cli agents.

This module handles bidirectional communication between the OpenAI Agent Builder
workflow (Aura Dev Suite) and the local aura-cli agent system via webhooks and
the OpenAI Responses API.
"""

import asyncio
import hashlib
import hmac
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WebhookEvent(BaseModel):
    """Represents an incoming webhook event from OpenAI."""
    event_type: str
    event_id: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = ""


class OpenAIConnectorConfig(BaseModel):
    """Configuration for the OpenAI connector."""
    enabled: bool = False
    project_id: str = ""
    workflow_id: str = ""
    webhook_url: str = ""
    webhook_events: list[str] = Field(default_factory=list)
    agent_mapping: dict[str, str] = Field(default_factory=dict)
    sync_on_events: list[str] = Field(default_factory=list)
    auto_dispatch: bool = False


class GitHubConnectorConfig(BaseModel):
    """Configuration for the GitHub connector."""
    enabled: bool = False
    repo: str = ""
    branch: str = "main"
    auto_pr: bool = False
    ci_workflows: list[str] = Field(default_factory=list)


class OpenAIConnector:
    """Bridges OpenAI Aura Dev Suite workflow with local aura-cli agents."""

    def __init__(self, config_path: str = "aura.config.json") -> None:
        self.config_path = Path(config_path)
        self.openai_config = OpenAIConnectorConfig()
        self.github_config = GitHubConnectorConfig()
        self._load_config()

    def _load_config(self) -> None:
        """Load connector configuration from aura.config.json."""
        if not self.config_path.exists():
            logger.warning("Config file not found: %s", self.config_path)
            return
        try:
            raw = json.loads(self.config_path.read_text())
            if "openai_connector" in raw:
                self.openai_config = OpenAIConnectorConfig(**raw["openai_connector"])
            if "github_connector" in raw:
                self.github_config = GitHubConnectorConfig(**raw["github_connector"])
            logger.info("Connector config loaded successfully")
        except Exception:
            logger.exception("Failed to load connector config")

    def verify_webhook_signature(
        self, payload: bytes, signature: str, secret: str
    ) -> bool:
        """Verify the HMAC-SHA256 signature of an incoming webhook."""
        expected = hmac.new(
            secret.encode(), payload, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(f"sha256={expected}", signature)

    async def handle_response(self, event: WebhookEvent) -> dict[str, Any]:
        """Handle a response.completed event from OpenAI."""
        logger.info("Handling response event: %s", event.event_id)
        result = event.data.get("output", {})
        category = result.get("category", "code_architecture")
        local_agent = self.openai_config.agent_mapping.get(category)
        if local_agent and self.openai_config.auto_dispatch:
            logger.info("Dispatching to local agent: %s", local_agent)
            return {"dispatched": True, "agent": local_agent, "category": category}
        return {"dispatched": False, "reason": "auto_dispatch disabled or no mapping"}

    async def handle_eval(self, event: WebhookEvent) -> dict[str, Any]:
        """Handle an eval.run.succeeded event from OpenAI."""
        logger.info("Handling eval event: %s", event.event_id)
        eval_results = event.data.get("results", [])
        return {
            "eval_processed": True,
            "results_count": len(eval_results),
            "event_id": event.event_id,
        }

    async def sync_to_openai(self, event: WebhookEvent) -> dict[str, Any]:
        """Sync local changes back to OpenAI workflow context."""
        logger.info("Syncing to OpenAI: %s", event.event_id)
        return {
            "synced": True,
            "workflow_id": self.openai_config.workflow_id,
            "project_id": self.openai_config.project_id,
        }

    async def dispatch_event(self, event: WebhookEvent) -> dict[str, Any]:
        """Route an incoming webhook event to the appropriate handler."""
        handlers = {
            "response.completed": self.handle_response,
            "response.failed": self._handle_failure,
            "eval.run.succeeded": self.handle_eval,
            "eval.run.failed": self._handle_failure,
            "batch.completed": self._handle_batch,
            "batch.failed": self._handle_failure,
        }
        handler = handlers.get(event.event_type)
        if handler:
            return await handler(event)
        logger.warning("No handler for event type: %s", event.event_type)
        return {"handled": False, "event_type": event.event_type}

    async def _handle_failure(self, event: WebhookEvent) -> dict[str, Any]:
        """Handle failure events with logging and optional retry."""
        logger.error("Event failed: %s - %s", event.event_type, event.event_id)
        error_info = event.data.get("error", {})
        return {
            "failed": True,
            "event_type": event.event_type,
            "error": error_info,
        }

    async def _handle_batch(self, event: WebhookEvent) -> dict[str, Any]:
        """Handle batch completion events."""
        logger.info("Batch completed: %s", event.event_id)
        batch_results = event.data.get("results", [])
        return {
            "batch_processed": True,
            "results_count": len(batch_results),
        }

    def get_agent_for_category(self, category: str) -> str | None:
        """Look up the local agent module for an OpenAI workflow category."""
        return self.openai_config.agent_mapping.get(category)

    def is_enabled(self) -> bool:
        """Check if the OpenAI connector is enabled."""
        return self.openai_config.enabled


# Module-level convenience functions for hook integration
_connector: OpenAIConnector | None = None


def _get_connector() -> OpenAIConnector:
    global _connector
    if _connector is None:
        _connector = OpenAIConnector()
    return _connector


async def handle_response_completed_hook(event_data: dict[str, Any]) -> dict[str, Any]:
    """Hook handler for on_response_completed events."""
    connector = _get_connector()
    event = WebhookEvent(
        event_type="response.completed",
        event_id=event_data.get("id", ""),
        data=event_data,
    )
    return await connector.handle_response(event)


async def handle_eval(event_data: dict[str, Any]) -> dict[str, Any]:
    """Hook handler for on_eval_succeeded events."""
    connector = _get_connector()
    event = WebhookEvent(
        event_type="eval.run.succeeded",
        event_id=event_data.get("id", ""),
        data=event_data,
    )
    return await connector.handle_eval(event)


async def sync_to_openai(event_data: dict[str, Any]) -> dict[str, Any]:
    """Hook handler for on_pr_merged events."""
    connector = _get_connector()
    event = WebhookEvent(
        event_type="sync",
        event_id=event_data.get("id", ""),
        data=event_data,
    )
    return await connector.sync_to_openai(event)
