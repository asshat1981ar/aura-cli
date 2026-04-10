"""Unit tests for agents/api_validation_agent.py.

The module uses `# ruff: noqa: F821` because all helper function calls are
intentional stubs — undefined names that act as pseudo-code placeholders.
We inject mock callables into the module's global namespace before calling
the function so the test exercises the real control flow without external deps.
"""

import importlib
import sys
import types
from unittest.mock import MagicMock


def _load_module_with_mocks():
    """Load the agent module and patch every undefined name with a MagicMock."""
    module_name = "agents.api_validation_agent"
    # Force fresh import so our patches take effect cleanly
    if module_name in sys.modules:
        del sys.modules[module_name]

    import agents.api_validation_agent as mod

    stub_names = [
        "analyze_api_contract_documentation",
        "prioritize_inconsistencies",
        "assess_current_api_validation_tools",
        "propose_tool_upgrades",
        "design_phased_rollout_steps",
        "define_validation_checkpoints",
        "predict_failure_modes_based_on_updates",
        "suggest_methods_for_failure_analysis",
        "design_feedback_loop_for_validation_results",
        "define_feedback_metrics",
        "engage_stakeholders_during_validation",
    ]
    for name in stub_names:
        setattr(mod, name, MagicMock(return_value=MagicMock()))

    return mod


class TestValidateApiContracts:
    def test_function_exists(self):
        import agents.api_validation_agent as mod

        assert hasattr(mod, "validate_api_contracts_after_feature_addition")
        assert callable(mod.validate_api_contracts_after_feature_addition)

    def test_returns_string_when_stubs_provided(self):
        mod = _load_module_with_mocks()
        result = mod.validate_api_contracts_after_feature_addition()
        assert isinstance(result, str)

    def test_return_value_signals_success(self):
        mod = _load_module_with_mocks()
        result = mod.validate_api_contracts_after_feature_addition()
        assert "successfully" in result.lower() or "established" in result.lower()

    def test_does_not_raise_with_mocks(self):
        mod = _load_module_with_mocks()
        # Should complete without any exception
        mod.validate_api_contracts_after_feature_addition()
