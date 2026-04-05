"""
Unit tests for aura_cli/dispatch.py

Tests for CLI dispatch logic, runtime resolution, and context preparation.
"""

import pytest
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from types import SimpleNamespace

# Import the module under test
from aura_cli.dispatch import (
    DispatchContext,
    DispatchRule,
    _resolve_dispatch_action,
    _resolve_beads_runtime_override,
    _resolve_runtime_mode,
    _prepare_runtime_context,
    _handle_help_dispatch,
    _handle_json_help_dispatch,
    _handle_doctor_dispatch,
    _handle_readiness_dispatch,
    _handle_bootstrap_dispatch,
    _handle_show_config_dispatch,
)


class TestDispatchContext:
    """Tests for DispatchContext dataclass."""
    
    def test_context_creation(self):
        """Test creating a DispatchContext."""
        parsed = SimpleNamespace(action="test")
        project_root = Path("/test")
        runtime_factory = Mock()
        args = SimpleNamespace()
        
        ctx = DispatchContext(
            parsed=parsed,
            project_root=project_root,
            runtime_factory=runtime_factory,
            args=args
        )
        
        assert ctx.parsed == parsed
        assert ctx.project_root == project_root
        assert ctx.runtime_factory == runtime_factory
        assert ctx.args == args
        assert ctx.runtime is None
    
    def test_context_with_runtime(self):
        """Test DispatchContext with runtime set."""
        ctx = DispatchContext(
            parsed=SimpleNamespace(),
            project_root=Path("/test"),
            runtime_factory=Mock(),
            args=SimpleNamespace(),
            runtime={"key": "value"}
        )
        
        assert ctx.runtime == {"key": "value"}


class TestDispatchRule:
    """Tests for DispatchRule dataclass."""
    
    def test_rule_creation(self):
        """Test creating a DispatchRule."""
        handler = Mock()
        rule = DispatchRule(
            action="test_action",
            requires_runtime=True,
            handler=handler
        )
        
        assert rule.action == "test_action"
        assert rule.requires_runtime is True
        assert rule.handler == handler
    
    def test_rule_immutable(self):
        """Test that DispatchRule is immutable (frozen)."""
        rule = DispatchRule(
            action="test",
            requires_runtime=False,
            handler=Mock()
        )
        
        with pytest.raises(AttributeError):
            rule.action = "modified"


class TestResolveDispatchAction:
    """Tests for _resolve_dispatch_action function."""
    
    def test_with_action_attribute(self):
        """Test resolving action when parsed has action attribute."""
        parsed = SimpleNamespace(action="goal_add")
        
        result = _resolve_dispatch_action(parsed)
        
        assert result == "goal_add"
    
    def test_without_action_attribute(self):
        """Test resolving action when parsed has no action attribute."""
        parsed = SimpleNamespace()
        
        result = _resolve_dispatch_action(parsed)
        
        assert result == "interactive"
    
    def test_with_none_action(self):
        """Test resolving action when action is None."""
        parsed = SimpleNamespace(action=None)
        
        result = _resolve_dispatch_action(parsed)
        
        assert result == "interactive"


class TestResolveBeadsRuntimeOverride:
    """Tests for _resolve_beads_runtime_override function."""
    
    def test_no_beads_args(self):
        """Test when no beads-related args are set."""
        args = SimpleNamespace()
        
        beads_config, beads_override = _resolve_beads_runtime_override(args)
        
        assert beads_config is None
        assert beads_override is None
    
    def test_beads_enabled(self):
        """Test with beads flag enabled."""
        args = SimpleNamespace(beads=True)
        
        beads_config, beads_override = _resolve_beads_runtime_override(args)
        
        assert beads_config is not None
        assert beads_config["enabled"] is True
        assert beads_override["source"] == "cli"
        assert beads_override["enabled"] is True
    
    def test_no_beads_disabled(self):
        """Test with no_beads flag."""
        args = SimpleNamespace(no_beads=True)
        
        beads_config, beads_override = _resolve_beads_runtime_override(args)
        
        assert beads_config["enabled"] is False
        assert beads_override["enabled"] is False
    
    def test_beads_required(self):
        """Test with beads_required flag."""
        args = SimpleNamespace(beads_required=True)
        
        beads_config, beads_override = _resolve_beads_runtime_override(args)
        
        assert beads_config["enabled"] is True
        assert beads_config["required"] is True
    
    def test_beads_optional(self):
        """Test with beads_optional flag."""
        args = SimpleNamespace(beads_optional=True)
        
        beads_config, beads_override = _resolve_beads_runtime_override(args)
        
        assert beads_config["enabled"] is True
        assert beads_config["required"] is False
    
    def test_multiple_beads_args_priority(self):
        """Test behavior with multiple beads args (last wins)."""
        args = SimpleNamespace(
            beads=True,
            no_beads=True  # This should override beads
        )
        
        beads_config, _ = _resolve_beads_runtime_override(args)
        
        # Both are processed, last one wins in the loop
        assert beads_config["enabled"] is False


class TestResolveRuntimeMode:
    """Tests for _resolve_runtime_mode function."""
    
    def test_goal_status_action(self):
        """Test runtime mode for goal_status action."""
        args = SimpleNamespace()
        
        result = _resolve_runtime_mode("goal_status", args)
        
        assert result == "queue"
    
    def test_goal_add_action(self):
        """Test runtime mode for goal_add action."""
        args = SimpleNamespace()
        
        result = _resolve_runtime_mode("goal_add", args)
        
        assert result == "queue"
    
    def test_interactive_action(self):
        """Test runtime mode for interactive action."""
        args = SimpleNamespace()
        
        result = _resolve_runtime_mode("interactive", args)
        
        assert result == "queue"
    
    def test_goal_once_with_dry_run(self):
        """Test runtime mode for goal_once with dry_run."""
        args = SimpleNamespace(dry_run=True)
        
        result = _resolve_runtime_mode("goal_once", args)
        
        assert result == "lean"
    
    def test_goal_once_without_dry_run(self):
        """Test runtime mode for goal_once without dry_run."""
        args = SimpleNamespace(dry_run=False)
        
        result = _resolve_runtime_mode("goal_once", args)
        
        assert result is None
    
    def test_unknown_action(self):
        """Test runtime mode for unknown action."""
        args = SimpleNamespace()
        
        result = _resolve_runtime_mode("unknown_action", args)
        
        assert result is None


class TestPrepareRuntimeContext:
    """Tests for _prepare_runtime_context function."""
    
    @pytest.fixture
    def mock_context(self):
        """Fixture providing a mock DispatchContext."""
        return DispatchContext(
            parsed=SimpleNamespace(action="test"),
            project_root=Path("/test"),
            runtime_factory=Mock(return_value={"runtime": "mock"}),
            args=SimpleNamespace(),
            runtime=None
        )
    
    @patch('aura_cli.dispatch._sync_cli_compat')
    @patch('aura_cli.dispatch._check_project_writability')
    @patch('aura_cli.dispatch.log_json')
    def test_prepare_with_existing_runtime(
        self, mock_log, mock_writable, mock_sync, mock_context
    ):
        """Test preparation when runtime already exists."""
        mock_context.runtime = {"existing": "runtime"}
        
        result = _prepare_runtime_context(mock_context)
        
        assert result is None
        mock_context.runtime_factory.assert_not_called()
    
    @patch('aura_cli.dispatch._sync_cli_compat')
    @patch('aura_cli.dispatch._check_project_writability')
    @patch('aura_cli.dispatch.log_json')
    def test_prepare_creates_runtime(
        self, mock_log, mock_writable, mock_sync, mock_context
    ):
        """Test that runtime is created when not existing."""
        mock_writable.return_value = True
        
        result = _prepare_runtime_context(mock_context)
        
        assert result is None
        mock_context.runtime_factory.assert_called_once()
        assert mock_context.runtime == {"runtime": "mock"}
    
    @patch('aura_cli.dispatch._sync_cli_compat')
    @patch('aura_cli.dispatch._check_project_writability')
    @patch('aura_cli.dispatch.log_json')
    def test_prepare_with_dry_run(
        self, mock_log, mock_writable, mock_sync, mock_context
    ):
        """Test preparation with dry_run flag."""
        mock_context.args = SimpleNamespace(dry_run=True)
        mock_writable.return_value = True
        
        _prepare_runtime_context(mock_context)
        
        call_args = mock_context.runtime_factory.call_args
        assert call_args[1]["overrides"]["dry_run"] is True
    
    @patch('aura_cli.dispatch._sync_cli_compat')
    @patch('aura_cli.dispatch._check_project_writability')
    @patch('aura_cli.dispatch.log_json')
    def test_prepare_with_model_override(
        self, mock_log, mock_writable, mock_sync, mock_context
    ):
        """Test preparation with model override."""
        mock_context.args = SimpleNamespace(model="claude-3-opus")
        mock_writable.return_value = True
        
        _prepare_runtime_context(mock_context)
        
        call_args = mock_context.runtime_factory.call_args
        assert call_args[1]["overrides"]["model_name"] == "claude-3-opus"
    
    @patch('aura_cli.dispatch._sync_cli_compat')
    @patch('aura_cli.dispatch._check_project_writability')
    @patch('aura_cli.dispatch.log_json')
    def test_prepare_not_writable(
        self, mock_log, mock_writable, mock_sync, mock_context
    ):
        """Test preparation when project is not writable."""
        mock_writable.return_value = False
        
        result = _prepare_runtime_context(mock_context)
        
        assert result == 1
        mock_log.assert_any_call("CRITICAL", "aura_cli_startup_aborted_not_writable")


class TestHandleHelpDispatch:
    """Tests for _handle_help_dispatch function."""
    
    @patch('aura_cli.dispatch.render_help')
    def test_help_dispatch_success(self, mock_render):
        """Test successful help dispatch."""
        mock_render.return_value = "Help text"
        ctx = DispatchContext(
            parsed=SimpleNamespace(),
            project_root=Path("/test"),
            runtime_factory=Mock(),
            args=SimpleNamespace(help_topics=None)
        )
        
        result = _handle_help_dispatch(ctx)
        
        assert result == 0
        mock_render.assert_called_once_with(None)
    
    @patch('aura_cli.dispatch.render_help')
    def test_help_dispatch_with_topics(self, mock_render):
        """Test help dispatch with specific topics."""
        mock_render.return_value = "Help for topic"
        ctx = DispatchContext(
            parsed=SimpleNamespace(),
            project_root=Path("/test"),
            runtime_factory=Mock(),
            args=SimpleNamespace(help_topics=["goal", "add"])
        )
        
        result = _handle_help_dispatch(ctx)
        
        assert result == 0
        mock_render.assert_called_once_with(["goal", "add"])
    
    @patch('aura_cli.dispatch.render_help')
    def test_help_dispatch_value_error(self, mock_render):
        """Test help dispatch with ValueError."""
        mock_render.side_effect = ValueError("Unknown topic")
        ctx = DispatchContext(
            parsed=SimpleNamespace(),
            project_root=Path("/test"),
            runtime_factory=Mock(),
            args=SimpleNamespace(json=False)
        )
        
        result = _handle_help_dispatch(ctx)
        
        assert result == 2
    
    @patch('aura_cli.dispatch.render_help')
    @patch('aura_cli.dispatch.attach_cli_warnings')
    @patch('aura_cli.dispatch.unknown_command_help_topic_payload')
    @patch('json.dumps')
    def test_help_dispatch_value_error_json(
        self, mock_json_dumps, mock_payload, mock_attach, mock_render
    ):
        """Test help dispatch ValueError with JSON output."""
        mock_render.side_effect = ValueError("Unknown topic")
        mock_payload.return_value = {"error": "unknown"}
        mock_attach.return_value = {"error": "unknown", "warning": "test"}
        
        ctx = DispatchContext(
            parsed=SimpleNamespace(),
            project_root=Path("/test"),
            runtime_factory=Mock(),
            args=SimpleNamespace(json=True)
        )
        
        result = _handle_help_dispatch(ctx)
        
        assert result == 2


class TestHandleJsonHelpDispatch:
    """Tests for _handle_json_help_dispatch function."""
    
    @patch('aura_cli.dispatch.render_help')
    def test_json_help_dispatch(self, mock_render):
        """Test JSON help dispatch."""
        mock_render.return_value = '{"help": "content"}'
        ctx = DispatchContext(
            parsed=SimpleNamespace(),
            project_root=Path("/test"),
            runtime_factory=Mock(),
            args=SimpleNamespace()
        )
        
        result = _handle_json_help_dispatch(ctx)
        
        assert result == 0
        mock_render.assert_called_once_with(format="json")


class TestHandleDoctorDispatch:
    """Tests for _handle_doctor_dispatch function."""
    
    @patch('aura_cli.dispatch._handle_doctor')
    def test_doctor_dispatch(self, mock_handle):
        """Test doctor dispatch."""
        ctx = DispatchContext(
            parsed=SimpleNamespace(),
            project_root=Path("/test"),
            runtime_factory=Mock(),
            args=SimpleNamespace()
        )
        
        result = _handle_doctor_dispatch(ctx)
        
        assert result == 0
        mock_handle.assert_called_once_with(Path("/test"))


class TestHandleReadinessDispatch:
    """Tests for _handle_readiness_dispatch function."""
    
    @patch('aura_cli.dispatch._handle_readiness')
    def test_readiness_dispatch(self, mock_handle):
        """Test readiness dispatch."""
        ctx = DispatchContext(
            parsed=SimpleNamespace(),
            project_root=Path("/test"),
            runtime_factory=Mock(),
            args=SimpleNamespace()
        )
        
        result = _handle_readiness_dispatch(ctx)
        
        assert result == 0
        mock_handle.assert_called_once()


class TestHandleBootstrapDispatch:
    """Tests for _handle_bootstrap_dispatch function."""
    
    @patch('aura_cli.dispatch.config')
    def test_bootstrap_dispatch(self, mock_config):
        """Test bootstrap dispatch."""
        ctx = DispatchContext(
            parsed=SimpleNamespace(),
            project_root=Path("/test"),
            runtime_factory=Mock(),
            args=SimpleNamespace()
        )
        
        result = _handle_bootstrap_dispatch(ctx)
        
        assert result == 0
        mock_config.interactive_bootstrap.assert_called_once()


class TestHandleShowConfigDispatch:
    """Tests for _handle_show_config_dispatch function."""
    
    @patch('aura_cli.dispatch.config')
    @patch('builtins.print')
    def test_show_config_dispatch(self, mock_print, mock_config):
        """Test show config dispatch."""
        mock_config.show_config.return_value = {"key": "value"}
        ctx = DispatchContext(
            parsed=SimpleNamespace(),
            project_root=Path("/test"),
            runtime_factory=Mock(),
            args=SimpleNamespace()
        )
        
        result = _handle_show_config_dispatch(ctx)
        
        assert result == 0
        mock_config.show_config.assert_called_once()


class TestIntegration:
    """Integration tests for dispatch module."""
    
    def test_dispatch_context_workflow(self):
        """Test complete dispatch context workflow."""
        # Create a realistic context
        parsed = SimpleNamespace(action="goal_add", dry_run=True)
        project_root = Path("/tmp/test_project")
        runtime_factory = Mock(return_value={"mode": "test"})
        args = SimpleNamespace(
            dry_run=True,
            model="test-model",
            beads=True
        )
        
        ctx = DispatchContext(
            parsed=parsed,
            project_root=project_root,
            runtime_factory=runtime_factory,
            args=args
        )
        
        # Resolve action
        action = _resolve_dispatch_action(parsed)
        assert action == "goal_add"
        
        # Resolve runtime mode
        mode = _resolve_runtime_mode(action, args)
        assert mode == "queue"
        
        # Resolve beads config
        beads_config, beads_override = _resolve_beads_runtime_override(args)
        assert beads_config is not None
        assert beads_config["enabled"] is True
    
    def test_dispatch_rule_creation(self):
        """Test creating and using dispatch rules."""
        def mock_handler(ctx):
            return 0
        
        rule = DispatchRule(
            action="test_action",
            requires_runtime=True,
            handler=mock_handler
        )
        
        assert rule.action == "test_action"
        assert rule.requires_runtime is True
        assert rule.handler is mock_handler
        
        # Test handler can be called
        ctx = DispatchContext(
            parsed=SimpleNamespace(),
            project_root=Path("/test"),
            runtime_factory=Mock(),
            args=SimpleNamespace()
        )
        result = rule.handler(ctx)
        assert result == 0
