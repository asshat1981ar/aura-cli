"""Tests for core.failure_router — FailureRouter and FailureAction."""
import unittest

from core.failure_router import FailureAction, FailureRouter


class TestFailureRouterSandbox(unittest.TestCase):
    """Sandbox failure routing."""

    def setUp(self):
        self.router = FailureRouter(max_act_retries=3)

    def test_sandbox_failure_returns_retry_act(self):
        action = self.router.route_failure(
            phase="sandbox",
            attempt=1,
            error="subprocess exited with code 1",
        )
        self.assertEqual(action, FailureAction.RETRY_ACT)

    def test_sandbox_max_retries_returns_replan(self):
        action = self.router.route_failure(
            phase="sandbox",
            attempt=3,  # >= max_act_retries
            error="subprocess exited with code 1",
        )
        self.assertEqual(action, FailureAction.REPLAN)

    def test_sandbox_below_max_retries_returns_retry_act(self):
        action = self.router.route_failure(
            phase="sandbox",
            attempt=2,
            error="assertion failed",
        )
        self.assertEqual(action, FailureAction.RETRY_ACT)


class TestFailureRouterVerify(unittest.TestCase):
    """Verify phase failure routing."""

    def setUp(self):
        self.router = FailureRouter(max_act_retries=3)

    def test_verify_failure_with_test_output_retry_act(self):
        action = self.router.route_failure(
            phase="verify",
            attempt=1,
            error="assertion error",
            verification_output="3 passed, 2 failed",
        )
        self.assertEqual(action, FailureAction.RETRY_ACT)

    def test_verify_failure_with_passing_tests_max_retries_replan(self):
        action = self.router.route_failure(
            phase="verify",
            attempt=3,
            error="assertion error",
            verification_output="3 passed, 2 failed",
        )
        self.assertEqual(action, FailureAction.REPLAN)

    def test_verify_failure_no_passing_tests_returns_replan(self):
        action = self.router.route_failure(
            phase="verify",
            attempt=1,
            error="all tests failed",
            verification_output="0 passed, 5 failed",
        )
        self.assertEqual(action, FailureAction.REPLAN)

    def test_verify_failure_no_output_returns_replan(self):
        action = self.router.route_failure(
            phase="verify",
            attempt=1,
            error="tests failed",
        )
        self.assertEqual(action, FailureAction.REPLAN)


class TestFailureRouterEnvironmental(unittest.TestCase):
    """Environmental / external error routing."""

    def setUp(self):
        self.router = FailureRouter(max_act_retries=3)

    def test_network_error_returns_skip(self):
        action = self.router.route_failure(
            phase="verify",
            attempt=1,
            error="network unreachable",
        )
        self.assertEqual(action, FailureAction.SKIP)

    def test_connection_error_returns_skip(self):
        action = self.router.route_failure(
            phase="sandbox",
            attempt=1,
            error="connection refused to remote host",
        )
        self.assertEqual(action, FailureAction.SKIP)

    def test_timeout_error_returns_skip(self):
        action = self.router.route_failure(
            phase="verify",
            attempt=2,
            error="request timed out after 30s",
        )
        self.assertEqual(action, FailureAction.SKIP)

    def test_dns_error_returns_skip(self):
        action = self.router.route_failure(
            phase="verify",
            attempt=1,
            error="dns resolution failed for api.example.com",
        )
        self.assertEqual(action, FailureAction.SKIP)

    def test_certificate_error_returns_skip(self):
        action = self.router.route_failure(
            phase="sandbox",
            attempt=1,
            error="ssl certificate verification failed",
        )
        self.assertEqual(action, FailureAction.SKIP)

    def test_environmental_takes_precedence_over_sandbox(self):
        """Environmental error should be SKIP even for sandbox phase."""
        action = self.router.route_failure(
            phase="sandbox",
            attempt=1,
            error="sandbox failed due to network timeout",
        )
        self.assertEqual(action, FailureAction.SKIP)


class TestFailureRouterDefault(unittest.TestCase):
    """Default routing for unknown phases or errors."""

    def setUp(self):
        self.router = FailureRouter(max_act_retries=3)

    def test_unknown_phase_returns_skip(self):
        action = self.router.route_failure(
            phase="unknown_phase",
            attempt=1,
            error="something went wrong",
        )
        self.assertEqual(action, FailureAction.SKIP)

    def test_default_max_retries(self):
        router = FailureRouter()
        # Should work with default max_act_retries
        action = router.route_failure(
            phase="sandbox",
            attempt=1,
            error="failed",
        )
        self.assertEqual(action, FailureAction.RETRY_ACT)


class TestFailureActionEnum(unittest.TestCase):
    """Ensure FailureAction enum has expected members."""

    def test_all_actions_exist(self):
        self.assertIn("RETRY_ACT", FailureAction.__members__)
        self.assertIn("REPLAN", FailureAction.__members__)
        self.assertIn("SKIP", FailureAction.__members__)
        self.assertIn("ABORT", FailureAction.__members__)

    def test_actions_are_distinct(self):
        actions = list(FailureAction)
        self.assertEqual(len(actions), len(set(actions)))


if __name__ == "__main__":
    unittest.main()
