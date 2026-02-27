from pathlib import Path
from unittest.mock import patch

from agents.mutator import MutatorAgent


def _replace_in_file_block(file_path: str, old_content: str, new_content: str) -> str:
    return (
        f"REPLACE_IN_FILE {file_path}\n"
        "---OLD_CONTENT_START---\n"
        f"{old_content}\n"
        "---OLD_CONTENT_END---\n"
        "---NEW_CONTENT_START---\n"
        f"{new_content}\n"
        "---NEW_CONTENT_END---\n"
    )


def test_mutator_replace_in_file_blocks_stale_mismatch_and_preserves_file(tmp_path: Path):
    (tmp_path / "core").mkdir(parents=True, exist_ok=True)
    target = tmp_path / "core" / "example.py"
    target.write_text("print('before')\n", encoding="utf-8")

    agent = MutatorAgent(tmp_path)
    proposal = _replace_in_file_block(
        "core/example.py",
        "stale snippet",
        "print('after')",
    )

    with patch("core.file_tools.recover_old_code_from_git", return_value=None), \
         patch("agents.mutator.log_json") as mock_log:
        agent.apply_mutation(proposal)

    assert target.read_text(encoding="utf-8") == "print('before')\n"

    events = [c.args[1] for c in mock_log.call_args_list if len(c.args) >= 2]
    assert "old_code_mismatch_overwrite_blocked" in events
    assert "mutator_replace_in_file_failed" in events
    assert "mutator_replace_in_file_success" not in events

    blocked_logs = [
        c for c in mock_log.call_args_list
        if len(c.args) >= 2 and c.args[1] == "old_code_mismatch_overwrite_blocked"
    ]
    assert blocked_logs[0].kwargs["details"]["policy"] == "explicit_overwrite_file_required"


def test_mutator_replace_in_file_succeeds_on_exact_match(tmp_path: Path):
    (tmp_path / "core").mkdir(parents=True, exist_ok=True)
    target = tmp_path / "core" / "example.py"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    agent = MutatorAgent(tmp_path)
    proposal = _replace_in_file_block(
        "core/example.py",
        "beta",
        "BETA",
    )

    with patch("agents.mutator.log_json") as mock_log:
        agent.apply_mutation(proposal)

    assert target.read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"

    events = [c.args[1] for c in mock_log.call_args_list if len(c.args) >= 2]
    assert "mutator_replace_in_file_success" in events
    assert "mutator_replace_in_file_failed" not in events
    assert "old_code_mismatch_overwrite_blocked" not in events
