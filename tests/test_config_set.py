"""Tests for `aura config set` CLI command.

Sprint 6 — s6-model-router-config
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import MagicMock, patch

from aura_cli.cli_options import parse_cli_args
import aura_cli.cli_main as cli_main


class TestConfigSetDispatch(unittest.TestCase):
    """Functional tests for `aura config set <key> <value>`."""

    def _dispatch(self, argv, *, runtime_factory=None):
        parsed = parse_cli_args(argv)
        rf = runtime_factory or MagicMock()
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli_main.dispatch_command(parsed, project_root=Path("."), runtime_factory=rf)
        return code, out.getvalue(), err.getvalue(), rf

    # ------------------------------------------------------------------
    # Basic parse / routing
    # ------------------------------------------------------------------

    def test_parse_config_set_yields_config_set_action(self):
        """parse_cli_args(['config', 'set', 'k', 'v']) → action == 'config_set'."""
        parsed = parse_cli_args(["config", "set", "max_iterations", "20"])
        self.assertEqual(parsed.action, "config_set")

    def test_config_set_positional_args_bound(self):
        """Positional config_key and config_value are bound to the namespace."""
        parsed = parse_cli_args(["config", "set", "max_iterations", "20"])
        self.assertEqual(parsed.namespace.config_key, "max_iterations")
        self.assertEqual(parsed.namespace.config_value, "20")

    def test_config_set_dotted_key_bound(self):
        """Dotted key paths are accepted as-is by the parser."""
        parsed = parse_cli_args(["config", "set", "model.code_generation", "google/gemini-2.5-pro"])
        self.assertEqual(parsed.namespace.config_key, "model.code_generation")
        self.assertEqual(parsed.namespace.config_value, "google/gemini-2.5-pro")

    # ------------------------------------------------------------------
    # Dispatch handler — model.<task> dotted path
    # ------------------------------------------------------------------

    def test_config_set_model_routing_writes_to_config_file(self):
        """config set model.code_generation <model> updates model_routing in JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "aura.config.json"
            config_path.write_text(json.dumps({"model_routing": {}}))

            from core.config_manager import ConfigManager

            real_cm = ConfigManager(config_file=str(config_path))

            with patch("aura_cli.dispatch.config", real_cm), patch("aura_cli.cli_main.config", real_cm):
                code, out, err, _ = self._dispatch(["config", "set", "model.code_generation", "google/gemini-2.5-pro"])

            self.assertEqual(code, 0, msg=f"stderr: {err}")
            self.assertIn("Set model.code_generation = google/gemini-2.5-pro", out)

            written = json.loads(config_path.read_text())
            self.assertEqual(
                written.get("model_routing", {}).get("code_generation"),
                "google/gemini-2.5-pro",
            )

    def test_config_set_flat_key_writes_to_config_file(self):
        """config set max_iterations 42 writes a flat key to the config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "aura.config.json"
            config_path.write_text(json.dumps({}))

            from core.config_manager import ConfigManager

            real_cm = ConfigManager(config_file=str(config_path))

            with patch("aura_cli.dispatch.config", real_cm), patch("aura_cli.cli_main.config", real_cm):
                code, out, err, _ = self._dispatch(["config", "set", "max_iterations", "42"])

            self.assertEqual(code, 0, msg=f"stderr: {err}")
            self.assertIn("Set max_iterations = 42", out)

            written = json.loads(config_path.read_text())
            self.assertEqual(written.get("max_iterations"), "42")

    def test_config_set_prints_confirmation(self):
        """Successful set prints 'Set <key> = <value>' to stdout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "aura.config.json"
            config_path.write_text(json.dumps({}))

            from core.config_manager import ConfigManager

            real_cm = ConfigManager(config_file=str(config_path))

            with patch("aura_cli.dispatch.config", real_cm), patch("aura_cli.cli_main.config", real_cm):
                code, out, _err, _ = self._dispatch(["config", "set", "model.planning", "openai/gpt-4o"])

            self.assertEqual(code, 0)
            self.assertIn("Set model.planning = openai/gpt-4o", out)

    # ------------------------------------------------------------------
    # Multiple task types via model.<task> paths
    # ------------------------------------------------------------------

    def test_config_set_model_planning(self):
        """config set model.planning <model> writes to model_routing.planning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "aura.config.json"
            config_path.write_text(json.dumps({"model_routing": {}}))

            from core.config_manager import ConfigManager

            real_cm = ConfigManager(config_file=str(config_path))

            with patch("aura_cli.dispatch.config", real_cm), patch("aura_cli.cli_main.config", real_cm):
                code, out, err, _ = self._dispatch(["config", "set", "model.planning", "anthropic/claude-3.5-sonnet"])

            self.assertEqual(code, 0, msg=f"stderr: {err}")
            written = json.loads(config_path.read_text())
            self.assertEqual(
                written.get("model_routing", {}).get("planning"),
                "anthropic/claude-3.5-sonnet",
            )

    def test_config_set_model_quality(self):
        """config set model.quality <model> writes to model_routing.quality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "aura.config.json"
            config_path.write_text(json.dumps({"model_routing": {}}))

            from core.config_manager import ConfigManager

            real_cm = ConfigManager(config_file=str(config_path))

            with patch("aura_cli.dispatch.config", real_cm), patch("aura_cli.cli_main.config", real_cm):
                code, out, err, _ = self._dispatch(["config", "set", "model.quality", "openai/o3"])

            self.assertEqual(code, 0, msg=f"stderr: {err}")
            written = json.loads(config_path.read_text())
            self.assertEqual(
                written.get("model_routing", {}).get("quality"),
                "openai/o3",
            )

    # ------------------------------------------------------------------
    # Error path
    # ------------------------------------------------------------------

    def test_config_set_error_when_save_fails(self):
        """When update_config raises, command returns exit code 1."""
        failing_cm = MagicMock()
        failing_cm.update_config.side_effect = Exception("disk full")

        # _sync_cli_compat may copy config from cli_main → patch both
        with patch("aura_cli.dispatch.config", failing_cm), patch("aura_cli.cli_main.config", failing_cm):
            code, _out, err, _ = self._dispatch(["config", "set", "model.code_generation", "any-model"])

        self.assertEqual(code, 1)
        self.assertIn("failed to save config", err)


if __name__ == "__main__":
    unittest.main()
