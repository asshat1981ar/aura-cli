"""Tests for core/a2a/agent_card.py — AgentCapability, AgentCard."""

import json
import pytest

from core.a2a.agent_card import AgentCapability, AgentCard


# ---------------------------------------------------------------------------
# AgentCapability
# ---------------------------------------------------------------------------

class TestAgentCapability:
    def test_name_and_description(self):
        cap = AgentCapability(name="my_cap", description="does something")
        assert cap.name == "my_cap"
        assert cap.description == "does something"

    def test_default_empty_schemas(self):
        cap = AgentCapability(name="cap", description="d")
        assert cap.input_schema == {}
        assert cap.output_schema == {}

    def test_custom_schemas(self):
        cap = AgentCapability(
            name="cap", description="d",
            input_schema={"type": "object"},
            output_schema={"type": "string"},
        )
        assert cap.input_schema["type"] == "object"
        assert cap.output_schema["type"] == "string"


# ---------------------------------------------------------------------------
# AgentCard defaults
# ---------------------------------------------------------------------------

class TestAgentCardDefaults:
    def test_default_name(self):
        card = AgentCard()
        assert card.name == "AURA CLI"

    def test_default_supported_protocols(self):
        card = AgentCard()
        assert "a2a/1.0" in card.supported_protocols
        assert "mcp/1.0" in card.supported_protocols

    def test_default_authentication_type(self):
        card = AgentCard()
        assert card.authentication["type"] == "bearer"

    def test_empty_capabilities_by_default(self):
        card = AgentCard()
        assert card.capabilities == []


# ---------------------------------------------------------------------------
# AgentCard.default()
# ---------------------------------------------------------------------------

class TestAgentCardDefaultFactory:
    def test_url_uses_host_and_port(self):
        card = AgentCard.default(host="192.168.1.1", port=9000)
        assert card.url == "http://192.168.1.1:9000"

    def test_default_url_localhost_8010(self):
        card = AgentCard.default()
        assert card.url == "http://localhost:8010"

    def test_default_has_five_capabilities(self):
        card = AgentCard.default()
        assert len(card.capabilities) == 5

    def test_capability_names(self):
        card = AgentCard.default()
        names = {c.name for c in card.capabilities}
        assert "code_generation" in names
        assert "code_review" in names
        assert "test_generation" in names
        assert "autonomous_goal" in names
        assert "plan_generation" in names

    def test_each_capability_has_schemas(self):
        card = AgentCard.default()
        for cap in card.capabilities:
            assert isinstance(cap.input_schema, dict)
            assert isinstance(cap.output_schema, dict)


# ---------------------------------------------------------------------------
# AgentCard.to_dict / to_json
# ---------------------------------------------------------------------------

class TestAgentCardSerialization:
    def test_to_dict_has_required_keys(self):
        card = AgentCard.default()
        d = card.to_dict()
        for key in ("name", "description", "version", "url", "capabilities"):
            assert key in d

    def test_to_dict_capabilities_are_dicts(self):
        card = AgentCard.default()
        caps = card.to_dict()["capabilities"]
        assert all(isinstance(c, dict) for c in caps)

    def test_to_json_is_valid_json(self):
        card = AgentCard.default()
        parsed = json.loads(card.to_json())
        assert parsed["name"] == "AURA CLI"

    def test_to_json_pretty_printed(self):
        card = AgentCard.default()
        j = card.to_json()
        assert "\n" in j  # indent=2 produces newlines


# ---------------------------------------------------------------------------
# AgentCard.from_dict
# ---------------------------------------------------------------------------

class TestAgentCardFromDict:
    def test_roundtrip(self):
        original = AgentCard.default()
        d = original.to_dict()
        restored = AgentCard.from_dict(d)
        assert restored.name == original.name
        assert restored.url == original.url
        assert len(restored.capabilities) == len(original.capabilities)

    def test_capabilities_restored_as_agent_capability(self):
        original = AgentCard.default()
        d = original.to_dict()
        restored = AgentCard.from_dict(d)
        assert all(isinstance(c, AgentCapability) for c in restored.capabilities)

    def test_no_capabilities_in_dict(self):
        card = AgentCard.from_dict({"name": "Test", "url": "http://x"})
        assert card.name == "Test"
        assert card.capabilities == []
