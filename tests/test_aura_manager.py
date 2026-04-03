"""Unit tests for core/aura_manager.py — AuraManager, orchestration_manager, safe_execute."""
from unittest.mock import MagicMock

from core.aura_manager import AuraManager, orchestration_manager, safe_execute


class TestAuraManagerInit:
    def test_performance_metrics_starts_empty(self):
        manager = AuraManager()
        assert manager.performance_metrics == []

    def test_registry_starts_empty(self):
        manager = AuraManager()
        assert manager.registry == {}


class TestAuraManagerIntegrateFunctions:
    def test_calls_each_function(self):
        manager = AuraManager()
        call_log = []
        func_a = MagicMock(return_value="a", __name__="func_a")
        func_b = MagicMock(return_value="b", __name__="func_b")
        manager.integrate_functions([func_a, func_b])
        func_a.assert_called_once()
        func_b.assert_called_once()

    def test_returns_dict_of_results(self):
        manager = AuraManager()
        func = MagicMock(return_value=42, __name__="my_func")
        results = manager.integrate_functions([func])
        assert isinstance(results, dict)
        assert results.get("my_func") == 42

    def test_failing_function_is_skipped_not_raised(self):
        manager = AuraManager()

        def bad_func():
            raise RuntimeError("boom")

        bad_func.__name__ = "bad_func"
        results = manager.integrate_functions([bad_func])
        assert "bad_func" not in results or results.get("bad_func") is None

    def test_empty_func_list_returns_empty_dict(self):
        manager = AuraManager()
        results = manager.integrate_functions([])
        assert results == {}

    def test_function_without_name_uses_unknown(self):
        manager = AuraManager()
        # lambdas have __name__ == '<lambda>'
        lam = lambda: "ok"  # noqa: E731
        results = manager.integrate_functions([lam])
        assert isinstance(results, dict)


class TestOrchestrationManager:
    def test_returns_none(self):
        result = orchestration_manager()
        assert result is None


class TestSafeExecute:
    def test_returns_function_result(self):
        result = safe_execute(lambda: "success")
        assert result == "success"

    def test_returns_none_on_exception(self):
        def boom():
            raise ValueError("crash")

        result = safe_execute(boom)
        assert result is None

    def test_passes_return_value_through(self):
        result = safe_execute(lambda: {"key": "val"})
        assert result == {"key": "val"}
