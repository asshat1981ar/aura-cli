"""Tests for environments/ package — config, manager, isolation, bootstrap."""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from environments.config import EnvironmentConfig
from environments.isolation import (
    cleanup_stale,
    create_secure_secrets_dir,
    create_workspace_tree,
    jail_path,
    teardown_workspace,
)
from environments.manager import EnvironmentManager


class TestEnvironmentConfig(unittest.TestCase):
    """Tests for EnvironmentConfig dataclass."""

    def test_from_name_creates_correct_paths(self):
        base = Path("/tmp/test_envs")
        env = EnvironmentConfig.from_name("gemini", "gemini-cli", base)
        self.assertEqual(env.name, "gemini")
        self.assertEqual(env.cli_type, "gemini-cli")
        self.assertEqual(env.workspace_root, base / "gemini")
        self.assertEqual(env.config_dir, base / "gemini" / "config")
        self.assertEqual(env.log_dir, base / "gemini" / "logs")
        self.assertEqual(env.temp_dir, base / "gemini" / "temp")
        self.assertEqual(env.secrets_dir, base / "gemini" / "secrets")
        self.assertEqual(env.deps_dir, base / "gemini" / "deps")

    def test_from_name_gemini_mcp_config(self):
        base = Path("/tmp/test_envs")
        env = EnvironmentConfig.from_name("gemini", "gemini-cli", base)
        self.assertEqual(env.mcp_config_path.name, "gemini_settings.json")

    def test_from_name_claude_mcp_config(self):
        base = Path("/tmp/test_envs")
        env = EnvironmentConfig.from_name("claude", "claude-code", base)
        self.assertEqual(env.mcp_config_path.name, "mcp.json")

    def test_from_name_codex_mcp_config(self):
        base = Path("/tmp/test_envs")
        env = EnvironmentConfig.from_name("codex", "codex-cli", base)
        self.assertEqual(env.mcp_config_path.name, "codex.mcp.config.json")

    def test_as_dict_serialization(self):
        base = Path("/tmp/test_envs")
        env = EnvironmentConfig.from_name("gemini", "gemini-cli", base)
        d = env.as_dict()
        self.assertEqual(d["name"], "gemini")
        self.assertEqual(d["cli_type"], "gemini-cli")
        self.assertIn("workspace_root", d)
        self.assertIn("port_range", d)

    def test_frozen_dataclass(self):
        base = Path("/tmp/test_envs")
        env = EnvironmentConfig.from_name("gemini", "gemini-cli", base)
        with self.assertRaises(AttributeError):
            env.name = "modified"

    def test_custom_port_range(self):
        base = Path("/tmp/test_envs")
        env = EnvironmentConfig.from_name(
            "gemini", "gemini-cli", base, port_range=(9000, 9010)
        )
        self.assertEqual(env.port_range, (9000, 9010))


class TestIsolation(unittest.TestCase):
    """Tests for environments/isolation.py."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_jail_path_valid(self):
        result = jail_path(Path("config/test.json"), self.tmpdir)
        self.assertTrue(str(result).startswith(str(self.tmpdir.resolve())))

    def test_jail_path_escape_raises(self):
        with self.assertRaises(ValueError):
            jail_path(Path("../../etc/passwd"), self.tmpdir)

    def test_jail_path_absolute_inside(self):
        inner = self.tmpdir / "subdir"
        inner.mkdir()
        result = jail_path(Path("subdir"), self.tmpdir)
        self.assertEqual(result, inner.resolve())

    def test_create_secure_secrets_dir(self):
        secrets = self.tmpdir / "secrets"
        result = create_secure_secrets_dir(secrets)
        self.assertTrue(result.is_dir())
        mode = oct(os.stat(str(secrets)).st_mode)[-3:]
        self.assertEqual(mode, "700")

    def test_create_workspace_tree(self):
        workspace = self.tmpdir / "test_ws"
        dirs = create_workspace_tree(workspace)
        self.assertTrue((workspace / "config").is_dir())
        self.assertTrue((workspace / "logs").is_dir())
        self.assertTrue((workspace / "temp").is_dir())
        self.assertTrue((workspace / "secrets").is_dir())
        self.assertTrue((workspace / "deps").is_dir())
        self.assertIn("config", dirs)

    def test_cleanup_stale_removes_old_files(self):
        workspace = self.tmpdir / "ws"
        create_workspace_tree(workspace)
        # Create an "old" file
        old_file = workspace / "temp" / "old.txt"
        old_file.write_text("old")
        # Set mtime to 48 hours ago
        old_time = os.path.getmtime(str(old_file)) - 48 * 3600
        os.utime(str(old_file), (old_time, old_time))
        # Create a "new" file
        new_file = workspace / "temp" / "new.txt"
        new_file.write_text("new")

        result = cleanup_stale(workspace, max_age_hours=24)
        self.assertEqual(result["temp"], 1)
        self.assertFalse(old_file.exists())
        self.assertTrue(new_file.exists())

    def test_teardown_workspace_preserves_secrets(self):
        workspace = self.tmpdir / "ws"
        create_workspace_tree(workspace)
        (workspace / "config" / "test.json").write_text("{}")

        result = teardown_workspace(workspace, preserve_secrets=True)
        self.assertEqual(result["status"], "cleaned")
        self.assertTrue((workspace / "secrets").is_dir())
        self.assertFalse((workspace / "config").exists())

    def test_teardown_workspace_full_removal(self):
        workspace = self.tmpdir / "ws"
        create_workspace_tree(workspace)

        result = teardown_workspace(workspace, preserve_secrets=False)
        self.assertEqual(result["status"], "removed")
        self.assertFalse(workspace.exists())


class TestEnvironmentManager(unittest.TestCase):
    """Tests for EnvironmentManager."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        # Create a minimal aura.config.json
        config = {"model_name": "test", "mcp_servers": {}}
        (self.tmpdir / "aura.config.json").write_text(json.dumps(config))
        self.mgr = EnvironmentManager(
            project_root=self.tmpdir,
            workspaces_dir=self.tmpdir / "workspaces",
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_bootstrap_creates_workspace(self):
        env = self.mgr.bootstrap_environment("gemini", "gemini-cli")
        self.assertEqual(env.name, "gemini")
        self.assertTrue(env.workspace_root.is_dir())
        self.assertTrue(env.config_dir.is_dir())
        self.assertTrue(env.secrets_dir.is_dir())

    def test_bootstrap_writes_config(self):
        env = self.mgr.bootstrap_environment("claude", "claude-code")
        config_file = env.config_dir / "aura.config.json"
        self.assertTrue(config_file.exists())
        config = json.loads(config_file.read_text())
        self.assertEqual(config["environment_name"], "claude")

    def test_list_environments(self):
        self.mgr.bootstrap_environment("gemini", "gemini-cli")
        self.mgr.bootstrap_environment("claude", "claude-code")
        envs = self.mgr.list_environments()
        self.assertEqual(len(envs), 2)
        names = {e.name for e in envs}
        self.assertEqual(names, {"gemini", "claude"})

    def test_get_environment(self):
        self.mgr.bootstrap_environment("codex", "codex-cli")
        env = self.mgr.get_environment("codex")
        self.assertIsNotNone(env)
        self.assertEqual(env.cli_type, "codex-cli")

    def test_get_environment_not_found(self):
        self.assertIsNone(self.mgr.get_environment("nonexistent"))

    def test_teardown_environment(self):
        self.mgr.bootstrap_environment("gemini", "gemini-cli")
        result = self.mgr.teardown_environment("gemini")
        self.assertIn(result["status"], ("cleaned", "removed"))
        self.assertIsNone(self.mgr.get_environment("gemini"))

    def test_environment_health(self):
        self.mgr.bootstrap_environment("gemini", "gemini-cli")
        health = self.mgr.environment_health("gemini")
        self.assertEqual(health["status"], "healthy")
        self.assertIn("directories", health)

    def test_environment_health_not_found(self):
        health = self.mgr.environment_health("nonexistent")
        self.assertEqual(health["status"], "not_found")

    def test_registry_persistence(self):
        self.mgr.bootstrap_environment("gemini", "gemini-cli")
        # Create a new manager instance — should reload from registry
        mgr2 = EnvironmentManager(
            project_root=self.tmpdir,
            workspaces_dir=self.tmpdir / "workspaces",
        )
        env = mgr2.get_environment("gemini")
        self.assertIsNotNone(env)
        self.assertEqual(env.cli_type, "gemini-cli")

    def test_auto_detect_cli_type(self):
        env = self.mgr.bootstrap_environment("gemini")
        self.assertEqual(env.cli_type, "gemini-cli")

    def test_cleanup_environment(self):
        self.mgr.bootstrap_environment("gemini", "gemini-cli")
        result = self.mgr.cleanup_environment("gemini", max_age_hours=0)
        self.assertIn("temp", result)


if __name__ == "__main__":
    unittest.main()
