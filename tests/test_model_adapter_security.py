"""Tests for subprocess prompt injection sanitization (issue #333)."""

from __future__ import annotations

import shlex
import subprocess
from unittest.mock import patch, MagicMock

import pytest


class TestLocalCommandProfileSanitization:
    """_call_local_command_profile must sanitize shell metacharacters in prompt."""

    def _make_adapter(self):
        from core.model_adapter import ModelAdapter

        return ModelAdapter.__new__(ModelAdapter)

    def test_shell_metacharacters_are_escaped_in_rendered_parts(self):
        """Shell metacharacters in prompt must not appear unquoted in rendered args."""
        adapter = self._make_adapter()
        profile = {"command": "echo {prompt}"}
        malicious_prompt = "hello; rm -rf / && echo pwned"

        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            m = MagicMock()
            m.stdout = "safe"
            return m

        with patch("subprocess.run", side_effect=fake_run):
            adapter._call_local_command_profile(profile, malicious_prompt)

        rendered_prompt_arg = captured["args"][1]  # echo <prompt>
        # The semicolon and ampersands must be quoted/escaped — they should not
        # appear as bare tokens that the shell could interpret.
        assert ";" not in rendered_prompt_arg or rendered_prompt_arg.startswith("'") or rendered_prompt_arg.startswith('"'), f"Prompt metacharacters not sanitized. Got: {rendered_prompt_arg!r}"
        # shlex.quote wraps in single quotes; the raw semicolon should be inside quotes
        expected = shlex.quote(malicious_prompt)
        assert rendered_prompt_arg == expected, f"Expected shlex.quote result {expected!r}, got {rendered_prompt_arg!r}"

    def test_backtick_injection_escaped(self):
        """Backtick command substitution must be neutralized."""
        adapter = self._make_adapter()
        profile = {"command": "mymodel {prompt}"}
        prompt_with_backtick = "`cat /etc/passwd`"

        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            m = MagicMock()
            m.stdout = "safe"
            return m

        with patch("subprocess.run", side_effect=fake_run):
            adapter._call_local_command_profile(profile, prompt_with_backtick)

        rendered = captured["args"][1]
        expected = shlex.quote(prompt_with_backtick)
        assert rendered == expected

    def test_dollar_substitution_escaped(self):
        """Dollar-sign variable substitution must be neutralized."""
        adapter = self._make_adapter()
        profile = {"command": "mymodel {prompt}"}
        prompt = "$(whoami) said hello"

        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            m = MagicMock()
            m.stdout = "safe"
            return m

        with patch("subprocess.run", side_effect=fake_run):
            adapter._call_local_command_profile(profile, prompt)

        rendered = captured["args"][1]
        expected = shlex.quote(prompt)
        assert rendered == expected

    def test_safe_prompt_passes_through_quoted(self):
        """A normal prompt should still work — just be shlex-quoted."""
        adapter = self._make_adapter()
        profile = {"command": "echo {prompt}"}
        safe_prompt = "What is the capital of France?"

        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            m = MagicMock()
            m.stdout = "Paris"
            return m

        with patch("subprocess.run", side_effect=fake_run):
            result = adapter._call_local_command_profile(profile, safe_prompt)

        rendered = captured["args"][1]
        # shlex.quote of a safe string wraps it in single quotes
        assert rendered == shlex.quote(safe_prompt)
        assert result == "Paris"

    def test_empty_prompt_handled(self):
        """Empty/None prompt must not raise and must produce an empty string."""
        adapter = self._make_adapter()
        profile = {"command": "echo {prompt}"}

        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            m = MagicMock()
            m.stdout = ""
            return m

        with patch("subprocess.run", side_effect=fake_run):
            adapter._call_local_command_profile(profile, "")

        rendered = captured["args"][1]
        # shlex.quote("") returns "''" — that is acceptable
        assert rendered == shlex.quote("") or rendered == ""

    def test_stdin_path_not_affected(self):
        """When prompt goes via stdin (no {prompt} in command), args are untouched."""
        adapter = self._make_adapter()
        profile = {"command": "mymodel --stdin"}
        malicious_prompt = "hello; rm -rf /"

        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            captured["input"] = kwargs.get("input")
            m = MagicMock()
            m.stdout = "ok"
            return m

        with patch("subprocess.run", side_effect=fake_run):
            adapter._call_local_command_profile(profile, malicious_prompt)

        # stdin path: prompt goes to input=, not into args
        assert captured["args"] == ["mymodel", "--stdin"]
        assert captured["input"] == malicious_prompt
