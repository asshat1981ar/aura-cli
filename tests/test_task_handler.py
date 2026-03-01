import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import core.task_handler as task_handler


class _FakeGoalQueue:
    def __init__(self, goals):
        self._goals = list(goals)

    def has_goals(self):
        return bool(self._goals)

    def next(self):
        return self._goals.pop(0)


class _FakeGoalArchive:
    def __init__(self):
        self.records = []

    def record(self, goal, score):
        self.records.append((goal, score))


class _FakeTaskManager:
    def __init__(self, *args, **kwargs):
        self.root_tasks = []

    def add_task(self, task):
        self.root_tasks.append(task)

    def save(self):
        return None


class TestTaskHandlerLoopControls(unittest.TestCase):
    def test_run_goals_loop_honors_args_max_cycles(self):
        args = SimpleNamespace(dry_run=True, max_cycles=1)
        queue = _FakeGoalQueue(["Respect max cycles"])
        archive = _FakeGoalArchive()
        loop = MagicMock()
        loop.run.return_value = "{}"
        loop.current_score = 1.23

        with tempfile.TemporaryDirectory() as td, \
             patch("core.task_handler.TaskManager", _FakeTaskManager), \
             patch("core.task_handler.log_json") as mock_log:
            task_handler.run_goals_loop(
                args,
                queue,
                loop,
                debugger_instance=None,
                planner_instance=None,
                goal_archive=archive,
                project_root=Path(td),
                decompose=False,
            )

        self.assertEqual(loop.run.call_count, 1)
        self.assertEqual(archive.records, [("Respect max cycles", 1.23)])

        cycle_limit_logs = [
            c for c in mock_log.call_args_list
            if len(c.args) >= 2 and c.args[1] == "cycle_limit_reached"
        ]
        self.assertEqual(len(cycle_limit_logs), 1)
        self.assertEqual(cycle_limit_logs[0].kwargs.get("details", {}).get("cycle_limit"), 1)

    def test_run_goals_loop_retries_once_with_grounding_hint_after_invalid_implement_path(self):
        args = SimpleNamespace(dry_run=True, max_cycles=5)
        queue = _FakeGoalQueue(["Retry invalid implement path with grounding hint"])
        archive = _FakeGoalArchive()
        loop = MagicMock()
        loop.run.side_effect = [
            json.dumps(
                {
                    "IMPLEMENT": {
                        "file_path": "cli_app.py",
                        "old_code": "old",
                        "new_code": "new",
                    }
                }
            ),
            json.dumps({"FINAL_STATUS": "ok"}),
        ]
        loop.current_score = 0.5

        with tempfile.TemporaryDirectory() as td, \
             patch("core.task_handler.TaskManager", _FakeTaskManager), \
             patch("core.task_handler.log_json") as mock_log:
            project_root = Path(td)
            (project_root / "aura_cli").mkdir(parents=True, exist_ok=True)
            (project_root / "aura_cli" / "cli_app_helper.py").write_text("# helper\n", encoding="utf-8")
            (project_root / "core").mkdir(parents=True, exist_ok=True)
            (project_root / "core" / "queue_retry.py").write_text("# queue retry\n", encoding="utf-8")
            task_handler.run_goals_loop(
                args,
                queue,
                loop,
                debugger_instance=None,
                planner_instance=None,
                goal_archive=archive,
                project_root=project_root,
                decompose=False,
            )

        self.assertEqual(loop.run.call_count, 2, "Invalid path should trigger one grounded retry")
        self.assertEqual(
            archive.records,
            [("Retry invalid implement path with grounding hint", 0.5)],
        )

        first_goal = loop.run.call_args_list[0].args[0]
        second_goal = loop.run.call_args_list[1].args[0]
        self.assertEqual(first_goal, "Retry invalid implement path with grounding hint")
        self.assertIn("GROUNDING_HINT", second_goal)
        self.assertIn("cli_app.py", second_goal)
        self.assertIn("file_not_found", second_goal)
        self.assertIn("Candidate existing files", second_goal)
        self.assertIn("aura_cli/cli_app_helper.py", second_goal)

        events = [c.args[1] for c in mock_log.call_args_list if len(c.args) >= 2]
        self.assertIn("invalid_implement_target_path", events)
        self.assertIn("grounding_retry_scheduled", events)
        self.assertIn("goal_completed", events)
        self.assertNotIn("goal_terminated_without_convergence", events)
        self.assertNotIn("replace_code_skipped", events)

        invalid_logs = [
            c for c in mock_log.call_args_list
            if len(c.args) >= 2 and c.args[1] == "invalid_implement_target_path"
        ]
        self.assertEqual(invalid_logs[0].kwargs.get("details", {}).get("reason"), "file_not_found")
        self.assertIn(
            "aura_cli/cli_app_helper.py",
            invalid_logs[0].kwargs.get("details", {}).get("candidate_files", []),
        )
        self.assertTrue(invalid_logs[0].kwargs.get("details", {}).get("retry_with_grounding_hint"))

    def test_run_goals_loop_terminates_when_invalid_implement_path_repeats_after_retry(self):
        args = SimpleNamespace(dry_run=True, max_cycles=5)
        queue = _FakeGoalQueue(["Terminate after repeated invalid path"])
        archive = _FakeGoalArchive()
        loop = MagicMock()
        loop.run.return_value = json.dumps(
            {
                "IMPLEMENT": {
                    "file_path": "cli_app.py",
                    "old_code": "old",
                    "new_code": "new",
                }
            }
        )
        loop.current_score = 0.0

        with tempfile.TemporaryDirectory() as td, \
             patch("core.task_handler.TaskManager", _FakeTaskManager), \
             patch("core.task_handler.log_json") as mock_log:
            task_handler.run_goals_loop(
                args,
                queue,
                loop,
                debugger_instance=None,
                planner_instance=None,
                goal_archive=archive,
                project_root=Path(td),
                decompose=False,
            )

        self.assertEqual(loop.run.call_count, 2, "One retry should be attempted before terminating")
        self.assertEqual(archive.records, [("Terminate after repeated invalid path", 0.0)])

        events = [c.args[1] for c in mock_log.call_args_list if len(c.args) >= 2]
        self.assertIn("grounding_retry_scheduled", events)
        self.assertIn("goal_terminated_without_convergence", events)
        self.assertNotIn("replace_code_skipped", events)

    def test_run_goals_loop_does_not_build_candidates_for_valid_path(self):
        args = SimpleNamespace(dry_run=True, max_cycles=1)
        queue = _FakeGoalQueue(["Valid path should skip candidate scan"])
        archive = _FakeGoalArchive()
        loop = MagicMock()
        loop.run.return_value = json.dumps(
            {
                "IMPLEMENT": {
                    "file_path": "core/existing.py",
                    "old_code": "old",
                    "new_code": "new",
                }
            }
        )
        loop.current_score = 0.0

        with tempfile.TemporaryDirectory() as td, \
             patch("core.task_handler.TaskManager", _FakeTaskManager), \
             patch("core.task_handler.log_json"), \
             patch("core.task_handler._candidate_existing_files") as mock_candidates:
            project_root = Path(td)
            (project_root / "core").mkdir(parents=True, exist_ok=True)
            (project_root / "core" / "existing.py").write_text("print('x')\n", encoding="utf-8")
            task_handler.run_goals_loop(
                args,
                queue,
                loop,
                debugger_instance=None,
                planner_instance=None,
                goal_archive=archive,
                project_root=project_root,
                decompose=False,
            )

        mock_candidates.assert_not_called()
        self.assertEqual(loop.run.call_count, 1)
        self.assertEqual(archive.records, [("Valid path should skip candidate scan", 0.0)])

    def test_candidate_existing_files_prefers_exact_basename_match(self):
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)
            (project_root / "run_aura.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            (project_root / "tests").mkdir(parents=True, exist_ok=True)
            (project_root / "tests" / "test_run_aura_wrapper.py").write_text("# test\n", encoding="utf-8")

            candidates = task_handler._candidate_existing_files(
                project_root,
                "scripts/run_aura.sh",
                "Update run_aura.sh wrapper help",
                limit=4,
            )

        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0], "run_aura.sh")

    def test_invalid_path_grounding_hint_surfaces_closest_exact_match(self):
        hint = task_handler._invalid_path_grounding_hint(
            "scripts/run_aura.sh",
            "file_not_found",
            ["run_aura.sh", "tests/test_run_aura_wrapper.py"],
        )

        self.assertIn("Closest existing match: run_aura.sh", hint)
        self.assertIn("Do not invent a new top-level directory", hint)

    def test_candidate_existing_files_prefers_exact_python_basename_match(self):
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)
            (project_root / "agents" / "skills").mkdir(parents=True, exist_ok=True)
            (project_root / "agents" / "skills" / "structural_analyzer.py").write_text(
                "# analyzer\n",
                encoding="utf-8",
            )
            (project_root / "agents" / "skills" / "dependency_analyzer.py").write_text(
                "# dependency\n",
                encoding="utf-8",
            )

            candidates = task_handler._candidate_existing_files(
                project_root,
                "analysis/structural_analyzer.py",
                "Run architecture validation after refactor",
                limit=4,
            )

        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0], "agents/skills/structural_analyzer.py")

    def test_allow_new_test_file_target_for_regression_goal(self):
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)
            (project_root / "tests").mkdir(parents=True, exist_ok=True)

            result = task_handler._allow_new_test_file_target(
                project_root,
                "tests/test_cli_loop.py",
                "Add interactive CLI loop regression tests",
                "",
                False,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.name, "test_cli_loop.py")

    def test_run_goals_loop_allows_new_test_file_target_without_retry(self):
        args = SimpleNamespace(dry_run=True, max_cycles=1)
        queue = _FakeGoalQueue(["Add interactive CLI loop regression tests"])
        archive = _FakeGoalArchive()
        loop = MagicMock()
        loop.run.return_value = json.dumps(
            {
                "IMPLEMENT": {
                    "file_path": "tests/test_cli_loop.py",
                    "old_code": "",
                    "new_code": "def test_placeholder():\n    assert True\n",
                    "overwrite_file": False,
                }
            }
        )
        loop.current_score = 0.0

        with tempfile.TemporaryDirectory() as td, \
             patch("core.task_handler.TaskManager", _FakeTaskManager), \
             patch("core.task_handler.log_json") as mock_log:
            project_root = Path(td)
            (project_root / "tests").mkdir(parents=True, exist_ok=True)
            task_handler.run_goals_loop(
                args,
                queue,
                loop,
                debugger_instance=None,
                planner_instance=None,
                goal_archive=archive,
                project_root=project_root,
                decompose=False,
            )

        self.assertEqual(loop.run.call_count, 1)
        events = [c.args[1] for c in mock_log.call_args_list if len(c.args) >= 2]
        self.assertIn("allowed_new_test_target", events)
        self.assertIn("replace_code_skipped", events)
        self.assertNotIn("invalid_implement_target_path", events)

    def test_symbol_index_cache_candidates_filter_invalid_and_stale_paths(self):
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td)
            (project_root / "memory").mkdir(parents=True, exist_ok=True)
            (project_root / "core").mkdir(parents=True, exist_ok=True)
            (project_root / "core" / "good_target.py").write_text("# ok\n", encoding="utf-8")

            payload = {
                "name_index": {
                    "target": [
                        {"file": "core/good_target.py", "line": 1, "type": "function"},
                        {"file": "../outside.py", "line": 1, "type": "function"},
                        {"file": "core/missing_target.py", "line": 1, "type": "function"},
                    ]
                }
            }
            (project_root / "memory" / "symbol_index.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            result = task_handler._candidate_files_from_symbol_index_cache(project_root, ["target"], limit=6)

        self.assertEqual(result, ["core/good_target.py"])

    def test_grounding_hint_is_cleared_after_valid_target_cycle(self):
        args = SimpleNamespace(dry_run=True, max_cycles=3)
        queue = _FakeGoalQueue(["Clear grounding hint after valid cycle"])
        archive = _FakeGoalArchive()
        loop = MagicMock()
        loop.run.side_effect = [
            json.dumps(
                {
                    "IMPLEMENT": {
                        "file_path": "cli_app.py",
                        "old_code": "old",
                        "new_code": "new",
                    }
                }
            ),
            json.dumps(
                {
                    "IMPLEMENT": {
                        "file_path": "core/existing.py",
                        "old_code": "old",
                        "new_code": "new",
                    }
                }
            ),
            json.dumps({"FINAL_STATUS": "ok"}),
        ]
        loop.current_score = 0.9

        with tempfile.TemporaryDirectory() as td, \
             patch("core.task_handler.TaskManager", _FakeTaskManager), \
             patch("core.task_handler.log_json"):
            project_root = Path(td)
            (project_root / "aura_cli").mkdir(parents=True, exist_ok=True)
            (project_root / "aura_cli" / "cli_app_helper.py").write_text("# helper\n", encoding="utf-8")
            (project_root / "core").mkdir(parents=True, exist_ok=True)
            (project_root / "core" / "existing.py").write_text("# existing\n", encoding="utf-8")
            task_handler.run_goals_loop(
                args,
                queue,
                loop,
                debugger_instance=None,
                planner_instance=None,
                goal_archive=archive,
                project_root=project_root,
                decompose=False,
            )

        self.assertEqual(loop.run.call_count, 3)
        first_goal = loop.run.call_args_list[0].args[0]
        second_goal = loop.run.call_args_list[1].args[0]
        third_goal = loop.run.call_args_list[2].args[0]
        self.assertEqual(first_goal, "Clear grounding hint after valid cycle")
        self.assertIn("GROUNDING_HINT", second_goal)
        self.assertEqual(third_goal, "Clear grounding hint after valid cycle")

    def test_run_goals_loop_retries_once_then_terminates_on_repeated_mismatch_overwrite_block(self):
        args = SimpleNamespace(dry_run=False, max_cycles=3)
        queue = _FakeGoalQueue(["Retry then terminate on repeated mismatch overwrite block"])
        archive = _FakeGoalArchive()
        loop = MagicMock()
        loop.run.return_value = json.dumps(
            {
                "IMPLEMENT": {
                    "file_path": "core/existing.py",
                    "old_code": "missing_old_code_marker()",
                    "new_code": "print('after')\n",
                }
            }
        )
        loop.current_score = 0.77

        with tempfile.TemporaryDirectory() as td, \
             patch("core.task_handler.TaskManager", _FakeTaskManager), \
             patch("core.task_handler.log_json") as mock_log, \
             patch("core.file_tools.recover_old_code_from_git", return_value=None), \
             patch("core.file_tools.log_json"):
            project_root = Path(td)
            (project_root / "core").mkdir(parents=True, exist_ok=True)
            target = project_root / "core" / "existing.py"
            target.write_text("print('before')\n", encoding="utf-8")

            task_handler.run_goals_loop(
                args,
                queue,
                loop,
                debugger_instance=None,
                planner_instance=None,
                goal_archive=archive,
                project_root=project_root,
                decompose=False,
            )

            final_contents = target.read_text(encoding="utf-8")

        self.assertEqual(loop.run.call_count, 2)
        self.assertEqual(final_contents, "print('before')\n")
        self.assertEqual(
            archive.records,
            [("Retry then terminate on repeated mismatch overwrite block", 0.77)],
        )

        first_goal = loop.run.call_args_list[0].args[0]
        second_goal = loop.run.call_args_list[1].args[0]
        self.assertEqual(first_goal, "Retry then terminate on repeated mismatch overwrite block")
        self.assertIn("GROUNDING_HINT", second_goal)
        self.assertIn("overwrite_file", second_goal)
        self.assertIn("old_code to an empty string", second_goal)
        self.assertIn("core/existing.py", second_goal)

        events = [c.args[1] for c in mock_log.call_args_list if len(c.args) >= 2]
        self.assertIn("applying_code_change", events)
        self.assertIn("old_code_mismatch_overwrite_blocked", events)
        self.assertNotIn("old_code_not_found", events)
        self.assertIn("grounding_retry_scheduled", events)
        self.assertIn("goal_terminated_without_convergence", events)
        self.assertNotIn("goal_completed", events)

        blocked_logs = [
            c for c in mock_log.call_args_list
            if len(c.args) >= 2 and c.args[1] == "old_code_mismatch_overwrite_blocked"
        ]
        self.assertEqual(
            blocked_logs[0].kwargs.get("details", {}).get("policy"),
            "explicit_overwrite_file_required",
        )

    def test_run_goals_loop_retries_with_grounding_hint_after_mismatch_overwrite_block_and_completes(self):
        args = SimpleNamespace(dry_run=False, max_cycles=3)
        queue = _FakeGoalQueue(["Retry after mismatch overwrite block and complete"])
        archive = _FakeGoalArchive()
        loop = MagicMock()
        loop.run.side_effect = [
            json.dumps(
                {
                    "IMPLEMENT": {
                        "file_path": "core/existing.py",
                        "old_code": "missing_old_code_marker()",
                        "new_code": "print('after')\n",
                    }
                }
            ),
            json.dumps({"FINAL_STATUS": "ok"}),
        ]
        loop.current_score = 0.81

        with tempfile.TemporaryDirectory() as td, \
             patch("core.task_handler.TaskManager", _FakeTaskManager), \
             patch("core.task_handler.log_json") as mock_log, \
             patch("core.file_tools.recover_old_code_from_git", return_value=None), \
             patch("core.file_tools.log_json"):
            project_root = Path(td)
            (project_root / "core").mkdir(parents=True, exist_ok=True)
            target = project_root / "core" / "existing.py"
            target.write_text("print('before')\n", encoding="utf-8")

            task_handler.run_goals_loop(
                args,
                queue,
                loop,
                debugger_instance=None,
                planner_instance=None,
                goal_archive=archive,
                project_root=project_root,
                decompose=False,
            )

            final_contents = target.read_text(encoding="utf-8")

        self.assertEqual(loop.run.call_count, 2)
        self.assertEqual(final_contents, "print('before')\n")
        self.assertEqual(archive.records, [("Retry after mismatch overwrite block and complete", 0.81)])

        second_goal = loop.run.call_args_list[1].args[0]
        self.assertIn("GROUNDING_HINT", second_goal)
        self.assertIn("overwrite_file", second_goal)
        self.assertIn("old_code to an empty string", second_goal)
        self.assertIn("core/existing.py", second_goal)

        events = [c.args[1] for c in mock_log.call_args_list if len(c.args) >= 2]
        self.assertIn("old_code_mismatch_overwrite_blocked", events)
        self.assertIn("grounding_retry_scheduled", events)
        self.assertIn("goal_completed", events)
        self.assertNotIn("goal_terminated_without_convergence", events)
        self.assertNotIn("old_code_not_found", events)

    def test_run_goals_loop_allows_mismatch_overwrite_when_explicit_flag_is_set(self):
        args = SimpleNamespace(dry_run=False, max_cycles=3)
        queue = _FakeGoalQueue(["Allow mismatch overwrite with explicit flag"])
        archive = _FakeGoalArchive()
        loop = MagicMock()
        loop.run.side_effect = [
            json.dumps(
                {
                    "IMPLEMENT": {
                        "file_path": "core/existing.py",
                        "old_code": "",
                        "new_code": "print('after')\n",
                        "overwrite_file": True,
                    }
                }
            ),
            json.dumps({"FINAL_STATUS": "ok"}),
        ]
        loop.current_score = 0.88

        with tempfile.TemporaryDirectory() as td, \
             patch("core.task_handler.TaskManager", _FakeTaskManager), \
             patch("core.task_handler.log_json") as mock_log, \
             patch("core.file_tools.recover_old_code_from_git", return_value=None), \
             patch("core.file_tools.log_json"):
            project_root = Path(td)
            (project_root / "core").mkdir(parents=True, exist_ok=True)
            target = project_root / "core" / "existing.py"
            target.write_text("print('before')\n", encoding="utf-8")

            task_handler.run_goals_loop(
                args,
                queue,
                loop,
                debugger_instance=None,
                planner_instance=None,
                goal_archive=archive,
                project_root=project_root,
                decompose=False,
            )

            final_contents = target.read_text(encoding="utf-8")

        self.assertEqual(loop.run.call_count, 2)
        self.assertEqual(final_contents, "print('after')\n")
        self.assertEqual(archive.records, [("Allow mismatch overwrite with explicit flag", 0.88)])

        events = [c.args[1] for c in mock_log.call_args_list if len(c.args) >= 2]
        self.assertIn("applying_code_change", events)
        self.assertIn("goal_completed", events)
        self.assertNotIn("old_code_not_found", events)
        self.assertNotIn("goal_terminated_without_convergence", events)

    def test_run_goals_loop_blocks_explicit_overwrite_flag_when_old_code_is_non_empty(self):
        args = SimpleNamespace(dry_run=False, max_cycles=2)
        queue = _FakeGoalQueue(["Block explicit overwrite when old_code is non-empty"])
        archive = _FakeGoalArchive()
        loop = MagicMock()
        loop.run.return_value = json.dumps(
            {
                "IMPLEMENT": {
                    "file_path": "core/existing.py",
                    "old_code": "stale marker",
                    "new_code": "print('after')\n",
                    "overwrite_file": True,
                }
            }
        )
        loop.current_score = 0.66

        with tempfile.TemporaryDirectory() as td, \
             patch("core.task_handler.TaskManager", _FakeTaskManager), \
             patch("core.task_handler.log_json") as mock_log, \
             patch("core.file_tools.recover_old_code_from_git", return_value=None), \
             patch("core.file_tools.log_json"):
            project_root = Path(td)
            (project_root / "core").mkdir(parents=True, exist_ok=True)
            target = project_root / "core" / "existing.py"
            target.write_text("print('before')\n", encoding="utf-8")

            task_handler.run_goals_loop(
                args,
                queue,
                loop,
                debugger_instance=None,
                planner_instance=None,
                goal_archive=archive,
                project_root=project_root,
                decompose=False,
            )

            final_contents = target.read_text(encoding="utf-8")

        self.assertEqual(loop.run.call_count, 2)
        self.assertEqual(final_contents, "print('before')\n")
        self.assertEqual(
            archive.records,
            [("Block explicit overwrite when old_code is non-empty", 0.66)],
        )

        second_goal = loop.run.call_args_list[1].args[0]
        self.assertIn("GROUNDING_HINT", second_goal)
        self.assertIn("old_code to an empty string", second_goal)
        self.assertIn("overwrite_file", second_goal)

        events = [c.args[1] for c in mock_log.call_args_list if len(c.args) >= 2]
        self.assertIn("old_code_mismatch_overwrite_blocked", events)
        self.assertIn("grounding_retry_scheduled", events)
        self.assertIn("goal_terminated_without_convergence", events)
        self.assertNotIn("goal_completed", events)


if __name__ == "__main__":
    unittest.main()
