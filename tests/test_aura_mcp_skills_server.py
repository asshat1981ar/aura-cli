from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

from agents.skills.registry import all_skills
import tools.aura_mcp_skills_server as mod


class _Response:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _DirectClient:
    def __init__(self, module):
        self._module = module

    def get(self, path: str) -> _Response:
        if path == "/health":
            return _Response(200, asyncio.run(self._module.health(None)))
        if path == "/discovery":
            return _Response(200, asyncio.run(self._module.discovery(None)))
        if path == "/tools":
            return _Response(200, asyncio.run(self._module.list_tools(None)))
        raise AssertionError(f"Unhandled GET path in test harness: {path}")


@pytest.fixture()
def client() -> _DirectClient:
    return _DirectClient(mod)


def test_tools_endpoint_covers_every_registered_skill(client: _DirectClient) -> None:
    response = client.get("/tools")
    assert response.status_code == 200

    payload = response.json()
    descriptors = {tool["name"]: tool for tool in payload["tools"]}
    registered = set(all_skills().keys())

    assert payload["count"] == len(registered)
    assert set(descriptors) == registered

    for name, descriptor in descriptors.items():
        assert descriptor["description"]
        assert descriptor["inputSchema"]["type"] == "object"
        assert isinstance(descriptor["inputSchema"]["properties"], dict)
        assert isinstance(descriptor["inputSchema"].get("required", []), list)
        if name == "structural_analyzer":
            assert "project_root" in descriptor["inputSchema"]["properties"]


class _FutureSkill:
    """Future skill docstring for descriptor inference."""

    name = "future_skill"


def test_descriptor_generation_falls_back_for_unmapped_skills(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_skills", {"future_skill": _FutureSkill()})

    descriptors = mod._list_skill_descriptors()

    assert descriptors == [
        {
            "name": "future_skill",
            "description": "Future skill docstring for descriptor inference.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }
    ]


def test_discovery_endpoint_exposes_current_server_and_registry(client: _DirectClient) -> None:
    response = client.get("/discovery")
    assert response.status_code == 200

    payload = response.json()
    assert payload["current_server"]["name"] == "aura-skills"
    assert payload["tool_count"] == len(mod._skills)
    assert any(server["name"] == "aura-copilot" for server in payload["servers"])
