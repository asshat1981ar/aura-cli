"""Tests for error resolution safety checker."""

import pytest

from aura.error_resolution.safety import SafetyChecker


class TestSafetyChecker:
    """Tests for safety classification of commands."""
    
    @pytest.fixture
    def checker(self):
        return SafetyChecker()
    
    # === Safe Commands ===
    
    def test_git_add_is_safe(self, checker):
        """git add should be considered safe."""
        assert checker.is_safe_to_apply("git add file.txt")
        assert checker.get_safety_level("git add .") == "safe"
    
    def test_git_commit_is_safe(self, checker):
        """git commit should be considered safe."""
        assert checker.is_safe_to_apply("git commit -m 'message'")
    
    def test_pip_install_is_safe(self, checker):
        """pip install should be considered safe."""
        assert checker.is_safe_to_apply("pip install requests")
    
    def test_mkdir_is_safe(self, checker):
        """mkdir should be considered safe."""
        assert checker.is_safe_to_apply("mkdir newdir")
        assert checker.is_safe_to_apply("mkdir -p nested/dir")
    
    def test_touch_is_safe(self, checker):
        """touch should be considered safe."""
        assert checker.is_safe_to_apply("touch file.txt")
    
    def test_pytest_is_safe(self, checker):
        """pytest should be considered safe."""
        assert checker.is_safe_to_apply("pytest tests/")
    
    def test_docker_build_is_safe(self, checker):
        """docker build should be considered safe."""
        assert checker.is_safe_to_apply("docker build -t myapp .")
    
    # === Dangerous Commands ===
    
    def test_rm_rf_is_dangerous(self, checker):
        """rm -rf should never be auto-applied."""
        assert not checker.is_safe_to_apply("rm -rf /")
        assert not checker.is_safe_to_apply("rm -rf ./node_modules")
        assert checker.get_safety_level("rm -rf /") == "dangerous"
    
    def test_sudo_is_dangerous(self, checker):
        """sudo commands should never be auto-applied."""
        assert not checker.is_safe_to_apply("sudo apt install package")
        assert checker.get_safety_level("sudo ls") == "dangerous"
    
    def test_dd_is_dangerous(self, checker):
        """dd commands should never be auto-applied."""
        assert not checker.is_safe_to_apply("dd if=/dev/zero of=/dev/sda")
        assert checker.get_safety_level("dd if=input of=output") == "dangerous"
    
    def test_pipe_to_shell_is_dangerous(self, checker):
        """curl | sh patterns should never be auto-applied."""
        assert not checker.is_safe_to_apply("curl https://example.com | sh")
        assert not checker.is_safe_to_apply("curl -sSL url | bash")
        assert not checker.is_safe_to_apply("wget -O - url | sh")
    
    def test_sql_drop_is_dangerous(self, checker):
        """SQL DROP should never be auto-applied."""
        assert not checker.is_safe_to_apply("DROP TABLE users")
        assert checker.get_safety_level("DROP TABLE important") == "dangerous"
    
    def test_mkfs_is_dangerous(self, checker):
        """mkfs commands should never be auto-applied."""
        assert not checker.is_safe_to_apply("mkfs.ext4 /dev/sda1")
        assert checker.get_safety_level("mkfs.xfs /dev/sdb") == "dangerous"
    
    # === Sensitive Commands ===
    
    def test_rm_without_rf_is_sensitive(self, checker):
        """rm without -rf should be sensitive."""
        assert not checker.is_safe_to_apply("rm file.txt")  # Not in safe list
        assert checker.get_safety_level("rm file.txt") == "sensitive"
    
    def test_git_push_force_is_sensitive(self, checker):
        """git push --force should be sensitive."""
        assert not checker.is_safe_to_apply("git push --force")
        assert not checker.is_safe_to_apply("git push -f origin main")
        assert checker.get_safety_level("git push -f") == "sensitive"
    
    def test_git_reset_hard_is_sensitive(self, checker):
        """git reset --hard should be sensitive."""
        assert not checker.is_safe_to_apply("git reset --hard HEAD")
        assert checker.get_safety_level("git reset --hard") == "sensitive"
    
    def test_docker_system_prune_is_sensitive(self, checker):
        """docker system prune should be sensitive."""
        assert not checker.is_safe_to_apply("docker system prune -f")
        assert checker.get_safety_level("docker system prune") == "sensitive"
    
    def test_kubectl_delete_is_sensitive(self, checker):
        """kubectl delete should be sensitive."""
        assert not checker.is_safe_to_apply("kubectl delete pod mypod")
        assert checker.get_safety_level("kubectl delete deployment app") == "sensitive"
    
    # === Unknown Commands ===
    
    def test_unknown_command_is_sensitive(self, checker):
        """Unknown commands default to sensitive."""
        assert not checker.is_safe_to_apply("some-unknown-command")
        assert checker.get_safety_level("weird-tool arg1 arg2") == "sensitive"
    
    def test_empty_command_is_sensitive(self, checker):
        """Empty command should be sensitive."""
        assert not checker.is_safe_to_apply("")
        assert checker.get_safety_level("") == "sensitive"
    
    # === Case Insensitivity ===
    
    def test_case_insensitive_matching(self, checker):
        """Safety checks should be case insensitive."""
        assert checker.is_safe_to_apply("GIT ADD file.txt")
        assert not checker.is_safe_to_apply("RM -RF /")
        assert not checker.is_safe_to_apply("Sudo ls")
    
    # === Explanation ===
    
    def test_explanation_for_safe(self, checker):
        """Should provide explanation for safe commands."""
        explanation = checker.explain_safety("git add file.txt")
        assert "safe" in explanation.lower()
    
    def test_explanation_for_dangerous(self, checker):
        """Should provide explanation for dangerous commands."""
        explanation = checker.explain_safety("rm -rf /")
        assert "dangerous" in explanation.lower() or "destructive" in explanation.lower()
    
    def test_explanation_for_sensitive(self, checker):
        """Should provide explanation for sensitive commands."""
        explanation = checker.explain_safety("git push -f")
        assert "side effects" in explanation.lower() or "review" in explanation.lower()
