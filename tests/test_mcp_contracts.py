"""Unit tests for core/mcp_contracts.py."""
from __future__ import annotations

import pytest

from core.mcp_contracts import (
    build_discovery_payload,
    build_health_payload,
    build_input_schema,
    build_tool_descriptor,
    build_tool_descriptors_from_schemas,
)


class TestBuildInputSchema:
    def test_empty_properties(self):
        schema = build_input_schema()
        assert schema["type"] == "object"
        assert schema["properties"] == {}
        assert schema["required"] == []

    def test_required_field_extracted(self):
        props = {"name": {"type": "string", "required": True}}
        schema = build_input_schema(props)
        assert "name" in schema["required"]

    def test_optional_field_not_in_required(self):
        props = {"name": {"type": "string", "required": False}}
        schema = build_input_schema(props)
        assert "name" not in schema["required"]

    def test_non_mapping_value_not_required(self):
        props = {"count": "integer"}
        schema = build_input_schema(props)
        assert "count" not in schema["required"]

    def test_multiple_required_fields(self):
        props = {
            "a": {"type": "string", "required": True},
            "b": {"type": "int", "required": True},
            "c": {"type": "bool"},
        }
        schema = build_input_schema(props)
        assert set(schema["required"]) == {"a", "b"}


class TestBuildToolDescriptor:
    def test_basic_descriptor(self):
        desc = build_tool_descriptor("my_tool", "Does something")
        assert desc["name"] == "my_tool"
        assert desc["description"] == "Does something"
        assert "inputSchema" in desc

    def test_input_properties_used_when_no_schema(self):
        props = {"query": {"type": "string", "required": True}}
        desc = build_tool_descriptor("search", "Search", input_properties=props)
        assert "query" in desc["inputSchema"]["properties"]
        assert "query" in desc["inputSchema"]["required"]

    def test_explicit_input_schema_overrides_properties(self):
        explicit = {"type": "object", "properties": {"x": {}}, "required": []}
        desc = build_tool_descriptor("t", "d", input_properties={"y": {}}, input_schema=explicit)
        assert "x" in desc["inputSchema"]["properties"]
        assert "y" not in desc["inputSchema"]["properties"]

    def test_extra_fields_included(self):
        desc = build_tool_descriptor("t", "d", version="1.0", tags=["a"])
        assert desc["version"] == "1.0"
        assert desc["tags"] == ["a"]


class TestBuildToolDescriptorsFromSchemas:
    def _schemas(self):
        return {
            "tool_a": {
                "description": "Tool A",
                "input": {"arg1": {"type": "string"}},
            },
            "tool_b": {
                "description": "Tool B",
                "inputSchema": {"type": "object", "properties": {}, "required": []},
            },
        }

    def test_returns_list_for_all_schemas(self):
        descs = build_tool_descriptors_from_schemas(self._schemas())
        assert len(descs) == 2

    def test_names_present(self):
        descs = build_tool_descriptors_from_schemas(self._schemas())
        names = {d["name"] for d in descs}
        assert "tool_a" in names
        assert "tool_b" in names

    def test_filter_by_names(self):
        descs = build_tool_descriptors_from_schemas(self._schemas(), names=["tool_a"])
        assert len(descs) == 1
        assert descs[0]["name"] == "tool_a"

    def test_uses_input_schema_when_present(self):
        descs = build_tool_descriptors_from_schemas(self._schemas())
        tool_b = next(d for d in descs if d["name"] == "tool_b")
        assert tool_b["inputSchema"]["type"] == "object"


class TestBuildHealthPayload:
    def test_basic_payload(self):
        payload = build_health_payload(server="aura-skills", version="1.2.3")
        assert payload["status"] == "ok"
        assert payload["server"] == "aura-skills"
        assert payload["version"] == "1.2.3"

    def test_tool_count_included_when_provided(self):
        payload = build_health_payload(server="s", version="v", tool_count=5)
        assert payload["tool_count"] == 5

    def test_tool_count_omitted_when_none(self):
        payload = build_health_payload(server="s", version="v")
        assert "tool_count" not in payload

    def test_custom_status(self):
        payload = build_health_payload(server="s", version="v", status="degraded")
        assert payload["status"] == "degraded"

    def test_extra_details_included(self):
        payload = build_health_payload(server="s", version="v", uptime=99)
        assert payload["uptime"] == 99


class TestBuildDiscoveryPayload:
    def test_basic_structure(self):
        payload = build_discovery_payload(
            current_server={"name": "me"},
            servers=[{"name": "other"}],
        )
        assert payload["discovery_version"] == "1.0.0"
        assert payload["current_server"] == {"name": "me"}
        assert len(payload["servers"]) == 1

    def test_custom_discovery_version(self):
        payload = build_discovery_payload(
            current_server={}, servers=[], discovery_version="2.0"
        )
        assert payload["discovery_version"] == "2.0"

    def test_extra_fields_included(self):
        payload = build_discovery_payload(
            current_server={}, servers=[], region="us-east-1"
        )
        assert payload["region"] == "us-east-1"

    def test_servers_copied_as_dicts(self):
        servers = [{"name": "a", "port": 8001}, {"name": "b", "port": 8002}]
        payload = build_discovery_payload(current_server={}, servers=servers)
        assert payload["servers"] == servers
