"""Tests for the 5 new agent types."""
import unittest
from unittest.mock import MagicMock, patch

from agents.python_agent import PythonAgentAdapter
from agents.typescript_agent import TypeScriptAgentAdapter
from agents.external_llm_agent import ExternalLLMAgentAdapter
from agents.monitoring_agent import MonitoringAgentAdapter
from agents.notification_agent import NotificationAgentAdapter
from agents.investigation_agent import InvestigationAgent
from agents.root_cause_analysis import RootCauseAnalysisAgent


class TestPythonAgentAdapter(unittest.TestCase):
    """Tests for PythonAgentAdapter."""

    def test_run_analyze_action(self):
        agent = PythonAgentAdapter()
        result = agent.run({"task": "Analyze code", "action": "analyze"})
        self.assertEqual(result["action"], "analyze")
        self.assertIn("lint_results", result)
        self.assertIn("type_check_results", result)
        self.assertIn("complexity", result)
        self.assertIn("coverage", result)

    def test_run_lint_action(self):
        agent = PythonAgentAdapter()
        result = agent.run({"task": "Lint code", "action": "lint"})
        self.assertEqual(result["action"], "lint")
        self.assertIn("lint_results", result)

    def test_run_generate_action_no_model(self):
        agent = PythonAgentAdapter()
        result = agent.run({"task": "Generate code", "action": "generate"})
        self.assertEqual(result["generated_code"], "")

    def test_run_with_skill(self):
        mock_skill = MagicMock()
        mock_skill.run.return_value = {"issues": []}
        agent = PythonAgentAdapter(skills={"linter_enforcer": mock_skill})
        result = agent.run({"task": "Lint", "action": "lint"})
        mock_skill.run.assert_called_once()

    def test_default_action_is_analyze(self):
        agent = PythonAgentAdapter()
        result = agent.run({"task": "Check code"})
        self.assertEqual(result["action"], "analyze")


class TestTypeScriptAgentAdapter(unittest.TestCase):
    """Tests for TypeScriptAgentAdapter."""

    def test_run_analyze_action(self):
        agent = TypeScriptAgentAdapter()
        with patch.object(agent, "_run_eslint", return_value={"status": "ok"}), \
             patch.object(agent, "_run_tsc", return_value={"status": "ok"}):
            result = agent.run({"task": "Analyze TS code", "action": "analyze"})
        self.assertEqual(result["action"], "analyze")
        self.assertIn("lint_results", result)
        self.assertIn("type_check_results", result)

    def test_run_generate_no_model(self):
        agent = TypeScriptAgentAdapter()
        result = agent.run({"task": "Generate component", "action": "generate"})
        self.assertEqual(result["generated_code"], "")

    def test_run_skill_not_available(self):
        agent = TypeScriptAgentAdapter()
        with patch.object(agent, "_run_eslint", return_value={"status": "ok"}), \
             patch.object(agent, "_run_tsc", return_value={"status": "ok"}):
            result = agent.run({"task": "Check API", "action": "analyze"})
        self.assertEqual(result["api_contract"]["status"], "skill_not_available")


class TestExternalLLMAgentAdapter(unittest.TestCase):
    """Tests for ExternalLLMAgentAdapter."""

    def test_run_no_model(self):
        agent = ExternalLLMAgentAdapter()
        result = agent.run({"task": "Generate text"})
        self.assertEqual(result["error"], "No model adapter configured")

    def test_run_with_model(self):
        mock_model = MagicMock(spec=[])
        mock_model.generate_for_task = MagicMock(return_value="generated text")
        agent = ExternalLLMAgentAdapter(model_adapter=mock_model)
        result = agent.run({"task": "Generate text", "category": "fast"})
        self.assertEqual(result["response"], "generated text")
        self.assertIsNone(result["error"])

    def test_run_with_context(self):
        mock_model = MagicMock(spec=[])
        mock_model.generate_for_task = MagicMock(return_value="result")
        agent = ExternalLLMAgentAdapter(model_adapter=mock_model)
        result = agent.run({
            "task": "Analyze",
            "context": "Some context",
            "category": "analyze",
        })
        call_args = mock_model.generate_for_task.call_args[0][1]
        self.assertIn("Some context", call_args)

    def test_run_handles_exception(self):
        mock_model = MagicMock(spec=[])
        mock_model.generate_for_task = MagicMock(side_effect=RuntimeError("API error"))
        agent = ExternalLLMAgentAdapter(model_adapter=mock_model)
        result = agent.run({"task": "Generate"})
        self.assertIn("API error", result["error"])


class TestMonitoringAgentAdapter(unittest.TestCase):
    """Tests for MonitoringAgentAdapter."""

    def test_status_action(self):
        agent = MonitoringAgentAdapter()
        with patch.object(agent, "_check_port", return_value=False):
            result = agent.run({"action": "status"})
        self.assertEqual(result["action"], "status")
        self.assertIn("overall", result)
        self.assertIn("servers", result)

    def test_scan_action(self):
        agent = MonitoringAgentAdapter()
        with patch.object(agent, "_check_port", return_value=False):
            result = agent.run({"action": "scan", "servers": {"fake": 59999}})
        self.assertEqual(result["action"], "scan")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["unhealthy"], 1)

    def test_alert_action(self):
        agent = MonitoringAgentAdapter()
        result = agent.run({"action": "alert"})
        self.assertEqual(result["action"], "alert")
        self.assertIn("thresholds", result)

    def test_query_action(self):
        agent = MonitoringAgentAdapter()
        agent.record_metric("test_metric", 42.0)
        result = agent.run({"action": "query", "metric_name": "test_metric"})
        self.assertEqual(result["count"], 1)

    def test_record_metric(self):
        agent = MonitoringAgentAdapter()
        agent.record_metric("cpu", 0.75, {"host": "server1"})
        self.assertEqual(len(agent._metric_history), 1)

    def test_default_action_is_status(self):
        agent = MonitoringAgentAdapter()
        with patch.object(agent, "_check_port", return_value=False):
            result = agent.run({})
        self.assertEqual(result["action"], "status")


class TestNotificationAgentAdapter(unittest.TestCase):
    """Tests for NotificationAgentAdapter."""

    def _failing_notification_post(self):
        return patch("requests.post", side_effect=RuntimeError("connection refused"))

    def test_missing_channel(self):
        agent = NotificationAgentAdapter()
        result = agent.run({"message": "test"})
        self.assertIn("error", result)

    def test_missing_message(self):
        agent = NotificationAgentAdapter()
        result = agent.run({"channel": "slack"})
        self.assertIn("error", result)

    def test_send_records_history(self):
        agent = NotificationAgentAdapter(notification_mcp_url="http://localhost:1")
        with self._failing_notification_post():
            result = agent.run({
                "channel": "webhook",
                "message": "test",
                "metadata": {"url": "http://example.com"},
            })
        # Will fail to connect but should still record in history
        history = agent.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["channel"], "webhook")

    def test_get_history_limit(self):
        agent = NotificationAgentAdapter(notification_mcp_url="http://localhost:1")
        with self._failing_notification_post():
            for i in range(5):
                agent.run({"channel": "webhook", "message": f"msg{i}", "metadata": {"url": "http://x"}})
        history = agent.get_history(limit=3)
        self.assertEqual(len(history), 3)

    def test_unknown_channel(self):
        agent = NotificationAgentAdapter(notification_mcp_url="http://localhost:1")
        with self._failing_notification_post():
            result = agent.run({"channel": "unknown", "message": "test"})
        self.assertEqual(result["status"], "failed")


class TestRootCauseAnalysisAgent(unittest.TestCase):
    """Tests for RootCauseAnalysisAgent."""

    def test_run_returns_structured_report(self):
        agent = RootCauseAnalysisAgent()
        result = agent.run(
            {
                "failures": ["SyntaxError: invalid syntax"],
                "logs": "Traceback ... SyntaxError: invalid syntax",
                "context": {"phase": "verify"},
            }
        )
        self.assertEqual(result["status"], "analyzed")
        self.assertIn("patterns", result)
        self.assertIn("recommended_actions", result)
        self.assertIn("syntax_error", result["patterns"])


class TestInvestigationAgent(unittest.TestCase):
    """Tests for InvestigationAgent."""

    def test_run_returns_structured_report(self):
        agent = InvestigationAgent()
        result = agent.run(
            {
                "goal": "Fix parser",
                "verification": {
                    "failures": ["SyntaxError: invalid syntax"],
                    "logs": "SyntaxError: invalid syntax",
                },
                "context": {"phase": "verify", "route": "plan"},
                "route": "plan",
            }
        )
        self.assertEqual(result["status"], "investigated")
        self.assertIn("verification_investigation", result)
        self.assertIn("remediation_plan", result)


class TestAgentRegistryIntegration(unittest.TestCase):
    """Test that new agents integrate with the registry."""

    def test_default_agents_includes_new_types(self):
        from unittest.mock import MagicMock
        brain = MagicMock()
        brain.remember = MagicMock()
        brain.recall_with_budget = MagicMock(return_value=[])
        brain.recall_weaknesses = MagicMock(return_value=[])

        from agents.registry import default_agents
        agents = default_agents(brain, "test-model")

        # Verify all specialized agents exist, including RCA support.
        expected = {
            "ingest", "plan", "critique", "synthesize", "act",
            "sandbox", "verify", "reflect",
            "python_agent", "typescript_agent", "external_llm",
            "monitoring", "notification", "telemetry", "self_correction",
            "code_search", "investigation", "root_cause_analysis",
        }
        self.assertEqual(set(agents.keys()), expected)

    def test_new_agents_have_run_method(self):
        from unittest.mock import MagicMock
        brain = MagicMock()
        brain.remember = MagicMock()
        brain.recall_with_budget = MagicMock(return_value=[])
        brain.recall_weaknesses = MagicMock(return_value=[])

        from agents.registry import default_agents
        agents = default_agents(brain, "test-model")

        for name in ["python_agent", "typescript_agent", "external_llm", "monitoring", "notification"]:
            self.assertTrue(
                hasattr(agents[name], "run"),
                f"Agent '{name}' missing run() method",
            )


if __name__ == "__main__":
    unittest.main()
