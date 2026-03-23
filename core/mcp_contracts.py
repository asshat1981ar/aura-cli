"""Shared helpers for MCP-style tool descriptors and service metadata."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping


def build_input_schema(properties: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Build a canonical MCP-style input schema from property metadata."""
    prop_dict = dict(properties or {})
    required = [key for key, value in prop_dict.items() if isinstance(value, Mapping) and value.get("required")]
    return {
        "type": "object",
        "properties": prop_dict,
        "required": required,
    }


def build_tool_descriptor(
    name: str,
    description: str,
    input_properties: Mapping[str, Any] | None = None,
    input_schema: Mapping[str, Any] | None = None,
    **extra_fields: Any,
) -> Dict[str, Any]:
    """Build a canonical MCP-style tool descriptor."""
    descriptor: Dict[str, Any] = {
        "name": name,
        "description": description,
        "inputSchema": dict(input_schema) if input_schema is not None else build_input_schema(input_properties),
    }
    descriptor.update(extra_fields)
    return descriptor


def build_tool_descriptors_from_schemas(
    schemas: Mapping[str, Mapping[str, Any]],
    *,
    names: Iterable[str] | None = None,
) -> list[Dict[str, Any]]:
    """Convert a tool schema mapping into canonical descriptors."""
    selected_names = list(names or schemas.keys())
    return [
        build_tool_descriptor(
            name=name,
            description=str(schemas[name]["description"]),
            input_properties=schemas[name].get("input"),
            input_schema=schemas[name].get("inputSchema"),
        )
        for name in selected_names
    ]


def build_health_payload(
    *,
    server: str,
    version: str,
    status: str = "ok",
    tool_count: int | None = None,
    **details: Any,
) -> Dict[str, Any]:
    """Build a standard health response payload."""
    payload: Dict[str, Any] = {
        "status": status,
        "server": server,
        "version": version,
    }
    if tool_count is not None:
        payload["tool_count"] = tool_count
    payload.update(details)
    return payload


def build_discovery_payload(
    *,
    current_server: Mapping[str, Any],
    servers: Iterable[Mapping[str, Any]],
    discovery_version: str = "1.0.0",
    **extra_fields: Any,
) -> Dict[str, Any]:
    """Build a service discovery payload for MCP-like servers."""
    payload: Dict[str, Any] = {
        "discovery_version": discovery_version,
        "current_server": dict(current_server),
        "servers": [dict(server) for server in servers],
    }
    payload.update(extra_fields)
    return payload
