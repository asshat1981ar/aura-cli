"""Tests for FailureRouter (#317)."""
import pytest
from core.failure_router import FailureRouter, FailureAction


class TestFailureRouter:
    """Test failure routing logic."""

    def test_sandbox_failure_returns_retry(self):
        """Code-level failures should retry act phase."""
        router = FailureRouter(max_act_retries=3)
        verification = {"failures": ["SyntaxError"], "logs": "indentation error"}
        action = router.route_failure(verification, attempt=1)
        assert action == FailureAction.RETRY_ACT

    def test_sandbox_failure_max_retries_returns_replan(self):
        """After max retries, should replan."""
        router = FailureRouter(max_act_retries=3)
        verification = {"failures": ["SyntaxError"], "logs": ""}
        action = router.route_failure(verification, attempt=3)
        assert action == FailureAction.REPLAN

    def test_structural_failure_returns_replan(self):
        """Architecture/design issues should trigger replan."""
        router = FailureRouter(max_act_retries=3)
        verification = {"failures": ["circular import"], "logs": ""}
        action = router.route_failure(verification, attempt=1)
        assert action == FailureAction.REPLAN

    def test_external_failure_returns_skip(self):
        """External/environmental issues should skip."""
        router = FailureRouter(max_act_retries=3)
        verification = {"failures": ["network timeout"], "logs": ""}
        action = router.route_failure(verification, attempt=1)
        assert action == FailureAction.SKIP

    def test_network_error_returns_skip(self):
        """Network errors should skip."""
        router = FailureRouter(max_act_retries=3)
        verification = {"failures": ["connection refused"], "logs": ""}
        action = router.route_failure(verification, attempt=1)
        assert action == FailureAction.SKIP

    def test_permission_error_returns_skip(self):
        """Permission errors should skip."""
        router = FailureRouter(max_act_retries=3)
        verification = {"failures": ["permission denied"], "logs": ""}
        action = router.route_failure(verification, attempt=1)
        assert action == FailureAction.SKIP

    def test_legacy_route_returns_strings(self):
        """Legacy route() method returns string literals."""
        router = FailureRouter(max_act_retries=3)
        
        result = router.route({"failures": ["error"], "logs": ""})
        assert result == "act"
        
        result = router.route({"failures": ["circular"], "logs": ""})
        assert result == "plan"
        
        result = router.route({"failures": ["network"], "logs": ""})
        assert result == "skip"
