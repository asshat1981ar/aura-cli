"""Tests for core/agent_sdk/subagent_definitions.py — SubagentDef, get_subagent_definitions."""

import pytest
from core.agent_sdk.subagent_definitions import (
    SubagentDef,
    get_subagent_definitions,
    get_subagent_for_task,
    _TASK_TYPE_MAP,
)


class TestSubagentDef:
    def test_fields(self):
        defn = SubagentDef(description="d", prompt="p", tools=["Read"])
        assert defn.description == "d"
        assert defn.prompt == "p"
        assert defn.tools == ["Read"]
        assert defn.model is None

    def test_model_override(self):
        defn = SubagentDef(description="d", prompt="p", tools=[], model="claude-opus-4-6")
        assert defn.model == "claude-opus-4-6"


class TestGetSubagentDefinitions:
    def test_returns_four_agents(self):
        defs = get_subagent_definitions()
        assert len(defs) == 4

    def test_contains_all_agent_names(self):
        defs = get_subagent_definitions()
        for name in ("planning-agent", "implementation-agent", "verification-agent", "research-agent"):
            assert name in defs

    def test_each_agent_has_tools(self):
        defs = get_subagent_definitions()
        for name, defn in defs.items():
            assert len(defn.tools) > 0, f"{name} has no tools"

    def test_each_agent_has_prompt(self):
        defs = get_subagent_definitions()
        for name, defn in defs.items():
            assert len(defn.prompt) > 20, f"{name} prompt too short"

    def test_implementation_agent_has_write(self):
        defs = get_subagent_definitions()
        assert "Write" in defs["implementation-agent"].tools

    def test_planning_agent_has_read(self):
        defs = get_subagent_definitions()
        assert "Read" in defs["planning-agent"].tools


class TestGetSubagentForTask:
    def test_plan_maps_to_planning_agent(self):
        assert get_subagent_for_task("plan") == "planning-agent"

    def test_planning_maps_to_planning_agent(self):
        assert get_subagent_for_task("planning") == "planning-agent"

    def test_code_maps_to_implementation_agent(self):
        assert get_subagent_for_task("code") == "implementation-agent"

    def test_implement_maps_to_implementation_agent(self):
        assert get_subagent_for_task("implement") == "implementation-agent"

    def test_test_maps_to_verification_agent(self):
        assert get_subagent_for_task("test") == "verification-agent"

    def test_research_maps_to_research_agent(self):
        assert get_subagent_for_task("research") == "research-agent"

    def test_unknown_returns_none(self):
        assert get_subagent_for_task("unknown_xyz") is None

    def test_case_insensitive(self):
        assert get_subagent_for_task("PLAN") == "planning-agent"
        assert get_subagent_for_task("Code") == "implementation-agent"

    def test_all_task_type_map_keys_resolve(self):
        for task_type in _TASK_TYPE_MAP:
            result = get_subagent_for_task(task_type)
            assert result is not None
