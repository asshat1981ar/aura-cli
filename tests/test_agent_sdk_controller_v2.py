# tests/test_agent_sdk_controller_v2.py
"""Tests for enhanced controller with production subsystems."""
import unittest
from pathlib import Path
from unittest.mock import MagicMock


class TestControllerV2Init(unittest.TestCase):
    """Test enhanced controller construction."""

    def test_accepts_new_optional_deps(self):
        from core.agent_sdk.controller import AuraController
        from core.agent_sdk.config import AgentSDKConfig
        config = AgentSDKConfig()
        controller = AuraController(
            config=config,
            project_root=Path("/tmp/test"),
            model_router=MagicMock(),
            workflow_executor=MagicMock(),
            session_store=MagicMock(),
            feedback=MagicMock(),
        )
        self.assertIsNotNone(controller.model_router)
        self.assertIsNotNone(controller.workflow_executor)
        self.assertIsNotNone(controller.session_store)
        self.assertIsNotNone(controller.feedback)

    def test_backward_compatible_without_new_deps(self):
        from core.agent_sdk.controller import AuraController
        from core.agent_sdk.config import AgentSDKConfig
        config = AgentSDKConfig()
        controller = AuraController(config=config, project_root=Path("/tmp/test"))
        self.assertIsNone(controller.model_router)
        self.assertIsNone(controller.session_store)


class TestControllerV2Options(unittest.TestCase):
    """Test enhanced _build_options."""

    def test_build_options_with_model_override(self):
        from core.agent_sdk.controller import AuraController
        from core.agent_sdk.config import AgentSDKConfig
        config = AgentSDKConfig()
        controller = AuraController(config=config, project_root=Path("/tmp/test"))
        opts = controller._build_options("Fix bug", model="claude-opus-4-6")
        self.assertEqual(opts.model, "claude-opus-4-6")

    def test_build_options_without_model_uses_config(self):
        from core.agent_sdk.controller import AuraController
        from core.agent_sdk.config import AgentSDKConfig
        config = AgentSDKConfig(model="claude-sonnet-4-6")
        controller = AuraController(config=config, project_root=Path("/tmp/test"))
        opts = controller._build_options("Fix bug")
        self.assertEqual(opts.model, "claude-sonnet-4-6")

    def test_context_builder_stored_as_attribute(self):
        from core.agent_sdk.controller import AuraController
        from core.agent_sdk.config import AgentSDKConfig
        config = AgentSDKConfig()
        controller = AuraController(config=config, project_root=Path("/tmp/test"))
        self.assertIsNotNone(controller.context_builder)


if __name__ == "__main__":
    unittest.main()
