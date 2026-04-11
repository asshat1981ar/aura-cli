"""Tests for REPL engine."""

import pytest
from unittest.mock import patch

from aura.shell.models import CommandResult, ShellCommand, CommandCategory
from aura.shell.repl import REPL


class TestREPL:
    @pytest.fixture
    def repl(self):
        return REPL(prompt="test> ", banner="Test Banner")
    
    @pytest.fixture
    def sample_command(self):
        return ShellCommand(
            name="hello",
            handler=lambda *args, **kwargs: CommandResult.ok(f"Hello {args[0] if args else 'World'}!"),
            category=CommandCategory.UTILITY,
        )
    
    def test_default_banner(self):
        repl = REPL()
        assert "AURA" in repl.banner
        assert "help" in repl.banner.lower()
    
    def test_register_command(self, repl, sample_command):
        repl.register_command(sample_command)
        
        assert "hello" in repl._commands
    
    def test_register_command_with_aliases(self, repl):
        cmd = ShellCommand(
            name="test",
            handler=lambda **kw: CommandResult.ok(),
            aliases=["t", "tst"],
        )
        repl.register_command(cmd)
        
        assert "test" in repl._commands
        assert "t" in repl._commands
        assert "tst" in repl._commands
    
    def test_parse_command_simple(self, repl):
        parts = repl._parse_command("hello world")
        assert parts == ["hello", "world"]
    
    def test_parse_command_with_quotes(self, repl):
        parts = repl._parse_command('echo "hello world"')
        assert parts == ["echo", "hello world"]
    
    def test_parse_command_empty(self, repl):
        parts = repl._parse_command("   ")
        assert parts == []
    
    def test_eval_known_command(self, repl, sample_command):
        repl.register_command(sample_command)
        result = repl._eval("hello Alice")
        
        assert result.success is True
        assert result.output == "Hello Alice!"
    
    def test_eval_unknown_command(self, repl):
        result = repl._eval("unknown")
        
        assert result.success is False
        assert "Unknown command" in result.error
    
    def test_eval_command_error(self, repl):
        def failing_handler(*args, **kwargs):
            raise ValueError("Test error")
        
        repl.register_command(ShellCommand(
            name="fail",
            handler=failing_handler,
        ))
        
        result = repl._eval("fail")
        
        assert result.success is False
        assert "Test error" in result.error
    
    def test_run_once(self, repl, sample_command):
        repl.register_command(sample_command)
        result = repl.run_once("hello Bob")
        
        assert result.success is True
        assert "Bob" in result.output
    
    def test_get_commands_by_category(self, repl):
        repl.register_commands([
            ShellCommand(name="cmd1", handler=lambda: None, category=CommandCategory.SYSTEM),
            ShellCommand(name="cmd2", handler=lambda: None, category=CommandCategory.SYSTEM),
            ShellCommand(name="cmd3", handler=lambda: None, category=CommandCategory.UTILITY),
        ])
        
        by_category = repl.get_commands_by_category()
        
        assert len(by_category[CommandCategory.SYSTEM]) == 2
        assert len(by_category[CommandCategory.UTILITY]) == 1
    
    def test_session_tracking(self, repl, sample_command):
        repl.register_command(sample_command)
        
        assert repl.session.command_count == 0
        
        repl._eval("hello")
        assert repl.session.command_count == 1
        
        repl._eval("hello")
        assert repl.session.command_count == 2
    
    def test_last_result_tracking(self, repl, sample_command):
        repl.register_command(sample_command)
        
        repl._eval("hello")
        
        assert repl.session.last_result is not None
        assert repl.session.last_result.success is True
