import json
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


def test_queue_loop_policy_block_retries_once_with_grounding_hint_and_no_overwrite(tmp_path: Path):
    args = SimpleNamespace(dry_run=False, max_cycles=3)
    queue = _FakeGoalQueue(["Integration policy-block retry"])
    archive = _FakeGoalArchive()
    loop = MagicMock()
    loop.run.side_effect = [
        json.dumps(
            {
                "IMPLEMENT": {
                    "file_path": "core/existing.py",
                    "old_code": "stale snippet",
                    "new_code": "print('after')\n",
                    "overwrite_file": True,
                }
            }
        ),
        json.dumps({"FINAL_STATUS": "ok"}),
    ]
    loop.current_score = 0.91

    (tmp_path / "core").mkdir(parents=True, exist_ok=True)
    target = tmp_path / "core" / "existing.py"
    target.write_text("print('before')\n", encoding="utf-8")

    with patch("core.task_handler.TaskManager", _FakeTaskManager), \
         patch("core.task_handler.log_json") as mock_log, \
         patch("core.file_tools.recover_old_code_from_git", return_value=None):
        task_handler.run_goals_loop(
            args,
            queue,
            loop,
            debugger_instance=None,
            planner_instance=None,
            goal_archive=archive,
            project_root=tmp_path,
            decompose=False,
        )

    assert target.read_text(encoding="utf-8") == "print('before')\n"
    assert loop.run.call_count == 2
    assert archive.records == [("Integration policy-block retry", 0.91)]

    second_goal = loop.run.call_args_list[1].args[0]
    assert "GROUNDING_HINT" in second_goal
    assert "overwrite_file" in second_goal
    assert "old_code to an empty string" in second_goal
    assert "core/existing.py" in second_goal

    events = [c.args[1] for c in mock_log.call_args_list if len(c.args) >= 2]
    assert "old_code_mismatch_overwrite_blocked" in events
    assert "grounding_retry_scheduled" in events
    assert "goal_completed" in events
    assert "old_code_not_found" not in events
    assert "goal_terminated_without_convergence" not in events

    blocked_logs = [
        c for c in mock_log.call_args_list
        if len(c.args) >= 2 and c.args[1] == "old_code_mismatch_overwrite_blocked"
    ]
    assert blocked_logs[0].kwargs["details"]["policy"] == "explicit_overwrite_file_required"
