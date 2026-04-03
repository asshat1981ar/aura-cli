"""Unit tests for core/mcp_architecture.py."""
from __future__ import annotations

import pytest

from core.mcp_architecture import (
    build_architecture_snapshot,
    default_observability_policy,
    default_routing_profile,
    supported_knowledge_backends,
)


class TestDefaultRoutingProfile:
    def test_returns_dict(self):
        profile = default_routing_profile()
        assert isinstance(profile, dict)

    def test_has_strategy(self):
        profile = default_routing_profile()
        assert "strategy" in profile
        assert isinstance(profile["strategy"], str)

    def test_has_service_order(self):
        profile = default_routing_profile()
        assert "service_order" in profile
        assert isinstance(profile["service_order"], list)
        assert len(profile["service_order"]) > 0

    def test_has_retry_policy(self):
        profile = default_routing_profile()
        assert "retry_policy" in profile
        rp = profile["retry_policy"]
        assert "max_attempts" in rp
        assert rp["max_attempts"] >= 1

    def test_has_stickiness(self):
        profile = default_routing_profile()
        assert "stickiness" in profile
        assert isinstance(profile["stickiness"], dict)

    def test_has_health_check_interval(self):
        profile = default_routing_profile()
        assert "health_check_interval_s" in profile
        assert profile["health_check_interval_s"] > 0


class TestDefaultObservabilityPolicy:
    def test_returns_dict(self):
        policy = default_observability_policy()
        assert isinstance(policy, dict)

    def test_has_logging_section(self):
        policy = default_observability_policy()
        assert "logging" in policy
        assert "format" in policy["logging"]

    def test_has_metrics_section(self):
        policy = default_observability_policy()
        assert "metrics" in policy
        assert "health_endpoint" in policy["metrics"]

    def test_has_security_section(self):
        policy = default_observability_policy()
        assert "security" in policy
        sec = policy["security"]
        assert "auth_mode" in sec
        assert "secret_source" in sec

    def test_audit_log_fields_is_list(self):
        policy = default_observability_policy()
        assert isinstance(policy["security"]["audit_log_fields"], list)


class TestSupportedKnowledgeBackends:
    def test_returns_list(self):
        backends = supported_knowledge_backends()
        assert isinstance(backends, list)
        assert len(backends) > 0

    def test_each_backend_has_name_and_role(self):
        for backend in supported_knowledge_backends():
            assert "name" in backend
            assert "role" in backend

    def test_each_backend_has_env_vars(self):
        for backend in supported_knowledge_backends():
            assert "env_vars" in backend
            assert isinstance(backend["env_vars"], list)

    def test_each_backend_has_stores(self):
        for backend in supported_knowledge_backends():
            assert "stores" in backend
            assert isinstance(backend["stores"], list)

    def test_neo4j_backend_present(self):
        names = [b["name"] for b in supported_knowledge_backends()]
        assert "neo4j" in names

    def test_sqlite_backend_present(self):
        names = [b["name"] for b in supported_knowledge_backends()]
        assert "sqlite-memory" in names


class TestBuildArchitectureSnapshot:
    def test_returns_dict(self):
        snapshot = build_architecture_snapshot()
        assert isinstance(snapshot, dict)

    def test_has_all_top_level_keys(self):
        snapshot = build_architecture_snapshot()
        assert "routing" in snapshot
        assert "observability_policy" in snapshot
        assert "knowledge_backends" in snapshot

    def test_routing_matches_default_profile(self):
        snapshot = build_architecture_snapshot()
        assert snapshot["routing"] == default_routing_profile()

    def test_observability_matches_default_policy(self):
        snapshot = build_architecture_snapshot()
        assert snapshot["observability_policy"] == default_observability_policy()

    def test_knowledge_backends_matches_supported(self):
        snapshot = build_architecture_snapshot()
        assert snapshot["knowledge_backends"] == supported_knowledge_backends()
