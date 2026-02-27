import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from git.exc import GitCommandError

from core.git_tools import GitTools, GitToolsError
from core.orchestrator import LoopOrchestrator
from core.policy import Policy
from memory.store import MemoryStore


class _StaticAgent:
    def __init__(self, payload):
        self._payload = payload

    def run(self, _input_data):
        return self._payload


def _base_agents(*, change_set):
    return {
        "ingest": _StaticAgent({
            "goal": "test goal",
            "snapshot": "snapshot",
            "memory_summary": "",
            "constraints": {},
        }),
        "plan": _StaticAgent({"steps": ["step"], "risks": []}),
        "critique": _StaticAgent({"issues": [], "fixes": []}),
        "synthesize": _StaticAgent({
            "tasks": [{"id": "t1", "title": "demo", "intent": "", "files": [], "tests": []}],
        }),
        "act": _StaticAgent(change_set),
        "sandbox": _StaticAgent({"passed": True, "summary": "ok"}),
        "verify": _StaticAgent({"status": "pass", "failures": [], "logs": ""}),
        "reflect": _StaticAgent({"summary": "done", "learnings": [], "next_actions": []}),
    }


class TestLoggingWorkflow(unittest.TestCase):
    def setUp(self):
        self.original_stdout = sys.stdout
        sys.stdout = io.StringIO()

    def tearDown(self):
        sys.stdout = self.original_stdout

    def _get_log_entries(self):
        output = sys.stdout.getvalue()
        entries = []
        for line in output.strip().split("\n"):
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return entries

    def _make_orchestrator(self, project_root: Path, *, change_set):
        return LoopOrchestrator(
            agents=_base_agents(change_set=change_set),
            memory_store=MemoryStore(project_root / "memory_store"),
            policy=Policy(max_cycles=1),
            project_root=project_root,
        )

    def test_orchestrator_logging_in_dry_run(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root / "test_file.py"
            orchestrator = self._make_orchestrator(
                root,
                change_set={
                    "changes": [{
                        "file_path": "test_file.py",
                        "old_code": "",
                        "new_code": "print('generated')\n",
                        "overwrite_file": True,
                    }],
                },
            )

            result = orchestrator.run_loop("Test logging dry run", max_cycles=1, dry_run=True)

            entries = self._get_log_entries()
            events = [e.get("event") for e in entries]
            self.assertIn("replace_code_skipped", events)
            self.assertNotIn("apply_change_failed", events)
            self.assertNotIn("old_code_not_found", events)
            self.assertEqual(result["stop_reason"], "PASS")
            self.assertFalse(target.exists())

    def test_orchestrator_logging_apply_mismatch_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root / "test_file.py"
            target.write_text("print('original')\n", encoding="utf-8")
            orchestrator = self._make_orchestrator(
                root,
                change_set={
                    "changes": [{
                        "file_path": "test_file.py",
                        "old_code": "missing snippet",
                        "new_code": "print('new')\n",
                        "overwrite_file": False,
                    }],
                },
            )

            with patch("core.file_tools.recover_old_code_from_git", return_value=None):
                result = orchestrator.run_loop("Test logging apply mismatch", max_cycles=1, dry_run=False)

            entries = self._get_log_entries()
            events = [e.get("event") for e in entries]
            self.assertIn("old_code_mismatch_overwrite_blocked", events)
            self.assertIn("verification_failed_routing", events)
            self.assertIn("verification_failure_external_skip", events)
            self.assertEqual(result["stop_reason"], "MAX_CYCLES")
            self.assertEqual(target.read_text(encoding="utf-8"), "print('original')\n")

    @patch("core.git_tools.Repo")
    def test_git_tools_logging_commit(self, mock_repo):
        sys.stdout = io.StringIO()

        repo = mock_repo.return_value
        repo.is_dirty.return_value = True
        repo.untracked_files = []
        repo.git.add.return_value = None
        repo.index.commit.return_value = None

        git_tools_instance = GitTools(repo_path=".")
        git_tools_instance.commit_all("Test Commit Message")
        log_entries = self._get_log_entries()

        self.assertTrue(any(e.get("event") == "git_committed" and e.get("level") == "INFO" for e in log_entries))
        self.assertTrue(any(e.get("details", {}).get("message") == "Test Commit Message" for e in log_entries))

    @patch("core.git_tools.Repo")
    def test_git_tools_logging_stash_pop_rollback_error(self, mock_repo):
        sys.stdout = io.StringIO()

        repo = mock_repo.return_value
        repo.is_dirty.return_value = True
        mock_git_cmd = MagicMock()
        repo.git = mock_git_cmd
        mock_git_cmd.stash.side_effect = GitCommandError("git stash", 1, "Simulated stash error output")
        repo.head.commit.parents = [MagicMock()]
        mock_git_cmd.reset.side_effect = GitCommandError("git reset", 1, "Simulated rollback error output")

        git_tools_instance = GitTools(repo_path=".")

        with self.assertRaises(GitToolsError):
            git_tools_instance.stash("Error stash")
        log_entries = self._get_log_entries()
        self.assertTrue(any(e.get("event") == "git_stash_failed" and e.get("level") == "ERROR" for e in log_entries))

        sys.stdout = io.StringIO()
        with self.assertRaises(GitToolsError):
            git_tools_instance.rollback_last_commit("Error rollback")
        log_entries = self._get_log_entries()
        self.assertTrue(any(e.get("event") == "git_rollback_failed" and e.get("level") == "ERROR" for e in log_entries))
