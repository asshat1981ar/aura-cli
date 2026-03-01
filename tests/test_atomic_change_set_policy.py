from pathlib import Path
from unittest.mock import patch

import pytest

from core.file_tools import AtomicChangeSet, MismatchOverwriteBlockedError


def test_atomic_change_set_blocks_stale_overwrite_and_restores_file(tmp_path: Path):
    target = tmp_path / "example.py"
    target.write_text("print('before')\n", encoding="utf-8")

    changes = [
        {
            "file_path": "example.py",
            "old_code": "stale snippet",
            "new_code": "print('after')\n",
            "overwrite_file": True,
        }
    ]

    with pytest.raises(MismatchOverwriteBlockedError), pytest.MonkeyPatch.context() as mp, \
         patch("core.file_tools.log_json") as mock_log:
        mp.setattr("core.file_tools.recover_old_code_from_git", lambda *args, **kwargs: None)
        AtomicChangeSet(changes, tmp_path).apply()

    assert target.read_text(encoding="utf-8") == "print('before')\n"
    events = [c.args[1] for c in mock_log.call_args_list if len(c.args) >= 2]
    assert "old_code_mismatch_overwrite_blocked" in events
    assert "atomic_change_set_failure" in events

    blocked_logs = [
        c for c in mock_log.call_args_list
        if len(c.args) >= 2 and c.args[1] == "old_code_mismatch_overwrite_blocked"
    ]
    assert blocked_logs[0].kwargs["details"]["policy"] == "explicit_overwrite_file_required"


def test_atomic_change_set_allows_explicit_full_file_overwrite(tmp_path: Path):
    target = tmp_path / "example.py"
    target.write_text("print('before')\n", encoding="utf-8")

    changes = [
        {
            "file_path": "example.py",
            "old_code": "",
            "new_code": "print('after')\n",
            "overwrite_file": True,
        }
    ]

    applied = AtomicChangeSet(changes, tmp_path).apply()

    assert applied == ["example.py"]
    assert target.read_text(encoding="utf-8") == "print('after')\n"
