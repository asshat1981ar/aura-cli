"""Unit tests for agents/technical_debt_agent.py — TechnicalDebtAgent."""
from agents.technical_debt_agent import TechnicalDebtAgent


class TestTechnicalDebtAgentInit:
    def test_hotspots_starts_empty(self):
        agent = TechnicalDebtAgent()
        assert agent.hotspots == []

    def test_exceptions_starts_empty(self):
        agent = TechnicalDebtAgent()
        assert agent.exceptions == []

    def test_secrets_starts_empty(self):
        agent = TechnicalDebtAgent()
        assert agent.secrets == []


class TestPrioritizeHotspots:
    def _sample_heatmap(self):
        return [
            {"file": "core/orchestrator.py", "failures": 10, "impact": 0.9},
            {"file": "agents/coder.py", "failures": 2, "impact": 0.5},
            {"file": "core/file_tools.py", "failures": 7, "impact": 0.8},
        ]

    def test_returns_list(self):
        agent = TechnicalDebtAgent()
        result = agent.prioritize_hotspots(self._sample_heatmap())
        assert isinstance(result, list)

    def test_caps_at_ten(self):
        agent = TechnicalDebtAgent()
        large = [{"file": f"f{i}.py", "failures": i, "impact": 0.5} for i in range(20)]
        result = agent.prioritize_hotspots(large)
        assert len(result) <= 10

    def test_sorted_by_failures_descending(self):
        agent = TechnicalDebtAgent()
        result = agent.prioritize_hotspots(self._sample_heatmap())
        failures = [h["failures"] for h in result]
        assert failures == sorted(failures, reverse=True)

    def test_stores_hotspots_on_instance(self):
        agent = TechnicalDebtAgent()
        agent.prioritize_hotspots(self._sample_heatmap())
        assert len(agent.hotspots) > 0


class TestExpandTestCoverage:
    def test_returns_string(self):
        agent = TechnicalDebtAgent()
        module = {"file": "core/goal_queue.py", "failures": 3, "impact": 0.6}
        result = agent.expand_test_coverage(module)
        assert isinstance(result, str)

    def test_result_references_module_file(self):
        agent = TechnicalDebtAgent()
        module = {"file": "core/goal_queue.py", "failures": 3, "impact": 0.6}
        result = agent.expand_test_coverage(module)
        assert "core/goal_queue.py" in result


class TestAuditExceptions:
    def test_returns_list(self):
        agent = TechnicalDebtAgent()
        logs = ["except Exception as e:", "    pass"]
        result = agent.audit_exceptions(logs)
        assert isinstance(result, list)

    def test_detects_generic_exception(self):
        agent = TechnicalDebtAgent()
        logs = ["try:", "    risky()", "except Exception:", "    pass"]
        result = agent.audit_exceptions(logs)
        assert len(result) > 0

    def test_detects_base_exception(self):
        agent = TechnicalDebtAgent()
        logs = ["except BaseException as e:"]
        result = agent.audit_exceptions(logs)
        assert len(result) > 0

    def test_ignores_clean_logs(self):
        agent = TechnicalDebtAgent()
        logs = ["try:", "    risky()", "except ValueError:", "    handle()"]
        result = agent.audit_exceptions(logs)
        assert result == []


class TestHardenTestEnv:
    def test_returns_string(self):
        agent = TechnicalDebtAgent()
        result = agent.harden_test_env()
        assert isinstance(result, str)

    def test_clears_secrets(self):
        agent = TechnicalDebtAgent()
        agent.secrets = ["DB_PASSWORD=hunter2"]
        agent.harden_test_env()
        assert agent.secrets == []


class TestTrackDebtMetrics:
    def test_returns_dict(self):
        agent = TechnicalDebtAgent()
        result = agent.track_debt_metrics(5, 3)
        assert isinstance(result, dict)

    def test_contains_expected_keys(self):
        agent = TechnicalDebtAgent()
        result = agent.track_debt_metrics(5, 3)
        assert "flake_reduction" in result
        assert "generic_exception_reduction" in result

    def test_values_match_inputs(self):
        agent = TechnicalDebtAgent()
        result = agent.track_debt_metrics(7, 2)
        assert result["flake_reduction"] == 7
        assert result["generic_exception_reduction"] == 2


class TestVisualizeHotspots:
    def test_does_not_raise(self, capsys):
        agent = TechnicalDebtAgent()
        data = [{"file": "core/orchestrator.py", "failures": 5, "impact": 0.8}]
        agent.visualize_hotspots(data)

    def test_prints_heatmap_header(self, capsys):
        agent = TechnicalDebtAgent()
        agent.visualize_hotspots([])
        captured = capsys.readouterr()
        assert "HEATMAP" in captured.out or "TECHNICAL DEBT" in captured.out

    def test_prints_high_risk_label(self, capsys):
        agent = TechnicalDebtAgent()
        data = [{"file": "core/orchestrator.py", "failures": 5, "impact": 0.9}]
        agent.visualize_hotspots(data)
        captured = capsys.readouterr()
        assert "HIGH" in captured.out
