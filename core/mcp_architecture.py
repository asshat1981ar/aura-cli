"""Architecture metadata for multi-server MCP deployments."""
from __future__ import annotations

from typing import Any, Dict


def default_routing_profile() -> Dict[str, Any]:
    return {
        "strategy": "health-aware-round-robin",
        "service_order": [
            "aura-dev-tools",
            "aura-skills",
            "aura-control",
            "aura-agentic-loop",
            "aura-copilot",
        ],
        "health_check_interval_s": 15,
        "retry_policy": {
            "max_attempts": 3,
            "backoff": "exponential-jitter",
            "retry_on": ["timeout", "connection_error", "503"],
        },
        "stickiness": {
            "goal_execution": "aura-dev-tools",
            "skill_execution": "aura-skills",
            "github_analysis": "aura-copilot",
        },
    }


def default_observability_policy() -> Dict[str, Any]:
    return {
        "logging": {
            "format": "structured-json",
            "event_field": "event",
            "correlation_id_header": "X-AURA-Request-ID",
        },
        "metrics": {
            "health_endpoint": "/health",
            "discovery_endpoint": "/discovery",
            "tool_latency_metric": "elapsed_ms",
        },
        "security": {
            "auth_mode": "bearer-token-env",
            "secret_source": "environment-variables",
            "audit_log_fields": ["server", "tool", "elapsed_ms", "error"],
        },
    }


def supported_knowledge_backends() -> list[Dict[str, Any]]:
    return [
        {
            "name": "neo4j",
            "role": "primary-knowledge-graph",
            "env_vars": ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"],
            "stores": ["agents", "tools", "capabilities", "execution-edges"],
        },
        {
            "name": "sqlite-memory",
            "role": "local-runtime-memory",
            "env_vars": [],
            "stores": ["brain", "response-cache", "semantic-memory"],
        },
        {
            "name": "external-api",
            "role": "supplemental-enrichment",
            "env_vars": ["AURA_API_KEY"],
            "stores": ["llm-results", "remote-context"],
        },
    ]


def build_architecture_snapshot() -> Dict[str, Any]:
    return {
        "routing": default_routing_profile(),
        "observability_policy": default_observability_policy(),
        "knowledge_backends": supported_knowledge_backends(),
    }
