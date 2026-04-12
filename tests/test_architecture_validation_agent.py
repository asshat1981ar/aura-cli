"""Unit tests for agents/architecture_validation_agent.py.

Like api_validation_agent, this module uses `# ruff: noqa: F821` — all helper
calls are intentional pseudo-code stubs. We inject MagicMock replacements into
the module namespace before invoking the real function.
"""

import sys
from unittest.mock import MagicMock


def _load_module_with_mocks():
    """Fresh-import the module and replace every undefined stub with a MagicMock."""
    module_name = "agents.architecture_validation_agent"
    if module_name in sys.modules:
        del sys.modules[module_name]

    import agents.architecture_validation_agent as mod

    stub_names = [
        "analyze_system_architecture",
        "compare_with_previous_architecture",
        "assess_dependency_management_tools",
        "identify_limitations",
        "propose_dependency_tool_upgrades",
        "integrate_with_CI_CD",
        "create_execution_plan",
        "establish_refactor_timeline",
        "design_feedback_loop",
        "predict_failure_modes",
        "define_validation_methods",
        "create_prioritization_framework",
        "suggest_architectural_improvements",
        "improve_documentation_and_communication",
        "engage_stakeholders_in_validation",
    ]
    for name in stub_names:
        setattr(mod, name, MagicMock(return_value=MagicMock()))

    return mod


class TestRunArchitectureValidation:
    def test_function_exists_and_is_callable(self):
        import agents.architecture_validation_agent as mod

        assert hasattr(mod, "run_architecture_validation_after_refactor")
        assert callable(mod.run_architecture_validation_after_refactor)

    def test_returns_string_when_stubs_provided(self):
        mod = _load_module_with_mocks()
        result = mod.run_architecture_validation_after_refactor()
        assert isinstance(result, str)

    def test_return_value_signals_success(self):
        mod = _load_module_with_mocks()
        result = mod.run_architecture_validation_after_refactor()
        assert "successfully" in result.lower() or "established" in result.lower()

    def test_does_not_raise_with_mocks(self):
        mod = _load_module_with_mocks()
        mod.run_architecture_validation_after_refactor()

    def test_return_contains_architecture_keyword(self):
        mod = _load_module_with_mocks()
        result = mod.run_architecture_validation_after_refactor()
        assert "architecture" in result.lower() or "validation" in result.lower()
