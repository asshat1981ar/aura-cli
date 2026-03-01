from pathlib import Path
from unittest.mock import patch

from core.file_tools import MismatchOverwriteBlockedError
from core.orchestrator import LoopOrchestrator
from core.policy import Policy
from memory.store import MemoryStore


def test_apply_change_set_logs_policy_block_for_mismatch_overwrite(tmp_path: Path):
    orchestrator = LoopOrchestrator(
        agents={},
        memory_store=MemoryStore(tmp_path / "mem"),
        policy=Policy(max_cycles=1),
        project_root=tmp_path,
    )

    change_set = {
        "file_path": "core/example.py",
        "old_code": "stale snippet",
        "new_code": "replacement",
        "overwrite_file": True,
    }

    with patch("core.orchestrator.apply_change_with_explicit_overwrite_policy", side_effect=MismatchOverwriteBlockedError("blocked")), \
         patch("core.orchestrator.log_json") as mock_log:
        result = orchestrator._apply_change_set(change_set, dry_run=False)

    assert result["applied"] == []
    assert result["failed"] == [{"file": "core/example.py", "error": "blocked"}]

    events = [c.args[1] for c in mock_log.call_args_list if len(c.args) >= 2]
    assert "old_code_mismatch_overwrite_blocked" in events
    assert "old_code_not_found" not in events

    blocked_logs = [
        c for c in mock_log.call_args_list
        if len(c.args) >= 2 and c.args[1] == "old_code_mismatch_overwrite_blocked"
    ]
    assert blocked_logs[0].kwargs["details"]["policy"] == "explicit_overwrite_file_required"


def test_apply_change_set_real_policy_blocks_stale_overwrite_and_preserves_file(tmp_path: Path):
    orchestrator = LoopOrchestrator(
        agents={},
        memory_store=MemoryStore(tmp_path / "mem"),
        policy=Policy(max_cycles=1),
        project_root=tmp_path,
    )

    (tmp_path / "core").mkdir(parents=True, exist_ok=True)
    target = tmp_path / "core" / "example.py"
    target.write_text("print('before')\n", encoding="utf-8")

    change_set = {
        "file_path": "core/example.py",
        "old_code": "stale snippet",
        "new_code": "print('after')\n",
        "overwrite_file": True,
    }

    with patch("core.file_tools.recover_old_code_from_git", return_value=None), \
         patch("core.orchestrator.log_json") as mock_log:
        result = orchestrator._apply_change_set(change_set, dry_run=False)

    assert target.read_text(encoding="utf-8") == "print('before')\n"
    assert result["applied"] == []
    assert result["failed"], "Expected failed apply result"
    assert result["failed"][0]["file"] == "core/example.py"
    assert "mismatch overwrite fallback is disabled" in result["failed"][0]["error"]

    events = [c.args[1] for c in mock_log.call_args_list if len(c.args) >= 2]
    assert "old_code_mismatch_overwrite_blocked" in events
    assert "old_code_not_found" not in events


def test_apply_change_set_real_policy_allows_explicit_full_file_overwrite(tmp_path: Path):
    orchestrator = LoopOrchestrator(
        agents={},
        memory_store=MemoryStore(tmp_path / "mem"),
        policy=Policy(max_cycles=1),
        project_root=tmp_path,
    )

    (tmp_path / "core").mkdir(parents=True, exist_ok=True)
    target = tmp_path / "core" / "example.py"
    target.write_text("print('before')\n", encoding="utf-8")

    change_set = {
        "file_path": "core/example.py",
        "old_code": "",
        "new_code": "print('after')\n",
        "overwrite_file": True,
    }

    with patch("core.file_tools.recover_old_code_from_git", return_value=None), \
         patch("core.orchestrator.log_json") as mock_log:
        result = orchestrator._apply_change_set(change_set, dry_run=False)

    assert target.read_text(encoding="utf-8") == "print('after')\n"
    assert result["applied"] == ["core/example.py"]
    assert result["failed"] == []

    events = [c.args[1] for c in mock_log.call_args_list if len(c.args) >= 2]
    assert "old_code_mismatch_overwrite_blocked" not in events
    assert "old_code_not_found" not in events
