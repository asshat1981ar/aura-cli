"""Tests for core/task_handler.py."""

import argparse
import json
import tempfile
import types
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from core.task_handler import (
    _check_project_writability,
    _goal_cycle_limit,
    _validate_change_target_path,
    _allow_new_test_file_target,
    _tokenize_for_path_matching,
    _normalize_cached_candidate_path,
    _candidate_files_from_symbol_index_cache,
    _candidate_files_by_exact_name,
    _candidate_files_from_repo_scan,
    _candidate_existing_files,
    _compose_loop_goal,
    _invalid_path_grounding_hint,
    _mismatch_overwrite_blocked_grounding_hint,
    _REPO_SCAN_SKIP_PARTS,
    run_goals_loop,
)


class TestCheckProjectWritability:
    """Test _check_project_writability function."""

    def test_writable_directory(self):
        """Test checking writable directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _check_project_writability(Path(tmpdir))
            assert result is True

    def test_non_writable_directory(self):
        """Test checking non-writable directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Make directory read-only
            Path(tmpdir).chmod(0o555)
            try:
                result = _check_project_writability(Path(tmpdir))
                assert result is False
            finally:
                Path(tmpdir).chmod(0o755)


class TestGoalCycleLimit:
    """Test _goal_cycle_limit function."""

    def test_default_limit(self):
        """Test default cycle limit."""
        args = argparse.Namespace()
        result = _goal_cycle_limit(args)
        assert result == 10

    def test_custom_limit(self):
        """Test custom cycle limit."""
        args = argparse.Namespace(max_cycles=5)
        result = _goal_cycle_limit(args)
        assert result == 5

    def test_invalid_limit_string(self):
        """Test invalid string limit defaults to 10."""
        args = argparse.Namespace(max_cycles="invalid")
        result = _goal_cycle_limit(args)
        assert result == 10

    def test_zero_limit_normalized(self):
        """Test zero limit is normalized to 1."""
        args = argparse.Namespace(max_cycles=0)
        result = _goal_cycle_limit(args)
        assert result == 1

    def test_negative_limit_normalized(self):
        """Test negative limit is normalized to 1."""
        args = argparse.Namespace(max_cycles=-5)
        result = _goal_cycle_limit(args)
        assert result == 1


class TestValidateChangeTargetPath:
    """Test _validate_change_target_path function."""

    def test_valid_file_path(self):
        """Test validating valid file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("content")

            target, error = _validate_change_target_path(Path(tmpdir), "test.py")

            assert target is not None
            assert error is None
            assert target.name == "test.py"

    def test_empty_path(self):
        """Test validating empty path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target, error = _validate_change_target_path(Path(tmpdir), "")

            assert target is None
            assert error == "missing_file_path"

    def test_path_outside_project(self):
        """Test path outside project root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target, error = _validate_change_target_path(Path(tmpdir), "../outside.py")

            assert target is None
            assert error == "outside_project_root"

    def test_nonexistent_file(self):
        """Test non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            target, error = _validate_change_target_path(Path(tmpdir), "nonexistent.py")

            assert target is None
            assert error == "file_not_found"


class TestAllowNewTestFileTarget:
    """Test _allow_new_test_file_target function."""

    def test_test_file_allowed(self):
        """Test test file is allowed to be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create tests directory and target file
            tests_dir = Path(tmpdir) / "tests"
            tests_dir.mkdir()
            target_file = tests_dir / "test_something.py"

            result = _allow_new_test_file_target(Path(tmpdir), "tests/test_something.py", "Create test for feature", "", None)
            assert result is not None
            assert result.name == "test_something.py"

    def test_non_test_file_rejected(self):
        """Test non-test file is not allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _allow_new_test_file_target(Path(tmpdir), "something.py", "Create something", "", None)
            assert result is None

    def test_tests_directory_allowed(self):
        """Test file in tests directory allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tests_dir = Path(tmpdir) / "tests"
            tests_dir.mkdir()
            target_file = tests_dir / "test_module.py"
            target_file.write_text("# test")

            result = _allow_new_test_file_target(Path(tmpdir), "tests/test_module.py", "Create regression test", "", None)
            assert result is not None


class TestTokenizeForPathMatching:
    """Test _tokenize_for_path_matching function."""

    def test_basic_tokenization(self):
        """Test basic text tokenization."""
        result = _tokenize_for_path_matching("Implement user authentication")
        assert "implement" not in result  # Stopword
        assert "user" in result
        assert "authentication" in result

    def test_empty_string(self):
        """Test empty string returns empty list."""
        result = _tokenize_for_path_matching("")
        assert result == []

    def test_only_stopwords(self):
        """Test string with only stopwords."""
        result = _tokenize_for_path_matching("the and for")
        assert result == []


class TestNormalizeCachedCandidatePath:
    """Test _normalize_cached_candidate_path function."""

    def test_valid_path_normalization(self):
        """Test valid path normalization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the file
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("content")

            result = _normalize_cached_candidate_path(Path(tmpdir), "test.py")
            assert result == "test.py"

    def test_path_outside_repo(self):
        """Test path outside repo returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _normalize_cached_candidate_path(Path(tmpdir), "/etc/passwd")
            assert result is None

    def test_skip_directories_filtered(self):
        """Test paths in skip directories are filtered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for skip_part in _REPO_SCAN_SKIP_PARTS:
                result = _normalize_cached_candidate_path(Path(tmpdir), f"{skip_part}/file.py")
                assert result is None


class TestCandidateFilesFromSymbolIndexCache:
    """Test _candidate_files_from_symbol_index_cache function."""

    def test_no_cache_files(self):
        """Test when no cache files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _candidate_files_from_symbol_index_cache(Path(tmpdir), ["query"], limit=6)
            assert result == []

    def test_cache_file_found(self):
        """Test finding candidates from cache file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir) / "memory"
            memory_dir.mkdir()

            # Create mock symbol index with proper structure
            # Also create the referenced files
            (Path(tmpdir) / "auth.py").write_text("# auth")

            symbol_index = {
                "name_index": {
                    "user_auth": [{"file": "auth.py"}, {"file": "models.py"}],
                    "data_model": [{"file": "models.py"}, {"file": "schema.py"}],
                }
            }
            (memory_dir / "symbol_index.json").write_text(json.dumps(symbol_index))

            result = _candidate_files_from_symbol_index_cache(Path(tmpdir), ["user", "auth"], limit=6)

            assert "auth.py" in result


class TestCandidateFilesByExactName:
    """Test _candidate_files_by_exact_name function."""

    def test_exact_match_found(self):
        """Test finding exact name match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("content")

            result = _candidate_files_by_exact_name(Path(tmpdir), "test.py", limit=6)

            assert "test.py" in result

    def test_exact_name_match(self):
        """Test exact name matching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "testfile.py"
            test_file.write_text("content")

            result = _candidate_files_by_exact_name(Path(tmpdir), "testfile.py", limit=6)

            assert "testfile.py" in result


class TestCandidateFilesFromRepoScan:
    """Test _candidate_files_from_repo_scan function."""

    def test_basic_scan(self):
        """Test basic repository scan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files
            (Path(tmpdir) / "auth.py").write_text("# auth")
            (Path(tmpdir) / "models.py").write_text("# models")

            result = _candidate_files_from_repo_scan(Path(tmpdir), "auth.py", ["auth"], limit=6)

            assert "auth.py" in result

    def test_skip_directories_excluded(self):
        """Test skip directories are excluded from scan."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files in skip directories
            pycache = Path(tmpdir) / "__pycache__"
            pycache.mkdir()
            (pycache / "cached.pyc").write_text("cached")

            # Create valid file
            (Path(tmpdir) / "valid.py").write_text("valid")

            result = _candidate_files_from_repo_scan(Path(tmpdir), "valid.py", ["valid"], limit=6)

            assert "valid.py" in result
            assert all("__pycache__" not in f for f in result)


class TestCandidateExistingFiles:
    """Test _candidate_existing_files function."""

    def test_candidates_returned(self):
        """Test candidates are returned for invalid path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files
            (Path(tmpdir) / "auth.py").write_text("# auth")
            (Path(tmpdir) / "user.py").write_text("# user")

            result = _candidate_existing_files(Path(tmpdir), "authentication.py", "Implement user auth", limit=6)

            assert isinstance(result, list)
            # Should find auth.py as candidate
            assert "auth.py" in result


class TestComposeLoopGoal:
    """Test _compose_loop_goal function."""

    def test_basic_composition(self):
        """Test basic goal composition."""
        result = _compose_loop_goal("Fix bug", None)
        assert result == "Fix bug"

    def test_with_grounding_hint(self):
        """Test composition with grounding hint."""
        result = _compose_loop_goal("Fix bug", "Hint: check line 42")
        assert "Fix bug" in result
        assert "Hint: check line 42" in result


class TestInvalidPathGroundingHint:
    """Test _invalid_path_grounding_hint function."""

    def test_hint_includes_candidates(self):
        """Test hint includes candidate files."""
        result = _invalid_path_grounding_hint("bad.py", "file_not_found", ["good.py", "better.py"])

        assert "bad.py" in result
        assert "good.py" in result
        assert "better.py" in result

    def test_hint_for_path_traversal(self):
        """Test hint for path traversal error."""
        result = _invalid_path_grounding_hint("../etc/passwd", "outside_project_root", [])

        assert "../etc/passwd" in result
        assert "outside_project_root" in result or "project root" in result.lower()


class TestMismatchOverwriteBlockedGroundingHint:
    """Test _mismatch_overwrite_blocked_grounding_hint function."""

    def test_hint_includes_file_path(self):
        """Test hint includes file path."""
        result = _mismatch_overwrite_blocked_grounding_hint("target.py")

        assert "target.py" in result
        assert "overwrite" in result.lower() or "mismatch" in result.lower()


# ---------------------------------------------------------------------------
# run_goals_loop helpers
# ---------------------------------------------------------------------------


def _make_args(**kwargs):
    defaults = {"max_cycles": 3, "dry_run": False}
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


def _make_goal_queue(goals):
    """Return a mock goal_queue that pops goals in order then returns empty."""
    remaining = list(goals)
    q = MagicMock()

    def has_goals():
        return bool(remaining)

    def next_goal():
        return remaining.pop(0)

    q.has_goals.side_effect = has_goals
    q.next.side_effect = next_goal
    return q


@patch("core.task_handler.log_json")
@patch("core.task_handler.InFlightTracker")
@patch("core.task_handler.TaskManager")
class TestRunGoalsLoop:
    """Tests for run_goals_loop."""

    def test_empty_queue_exits_without_processing(self, MockTM, MockIFT, mock_log):
        q = _make_goal_queue([])
        orch = MagicMock()
        archive = MagicMock()
        run_goals_loop(_make_args(), q, orch, None, None, archive, Path("/tmp"))
        orch.run_loop.assert_not_called()

    def test_single_goal_pass_marks_completed(self, MockTM, MockIFT, mock_log):
        orch = MagicMock()
        orch.run_loop.return_value = {"stop_reason": "PASS", "history": ["h"]}
        q = _make_goal_queue(["implement X"])
        archive = MagicMock()
        run_goals_loop(_make_args(), q, orch, None, None, archive, Path("/tmp"))
        orch.run_loop.assert_called_once()
        archive.record.assert_called_once()
        _, score = archive.record.call_args[0]
        assert score == 1.0

    def test_goal_max_cycles_scores_zero(self, MockTM, MockIFT, mock_log):
        orch = MagicMock()
        orch.run_loop.return_value = {"stop_reason": "MAX_CYCLES", "history": ["h"]}
        q = _make_goal_queue(["run task"])
        archive = MagicMock()
        run_goals_loop(_make_args(), q, orch, None, None, archive, Path("/tmp"))
        _, score = archive.record.call_args[0]
        assert score == 0.0

    def test_empty_history_scores_zero_even_on_pass(self, MockTM, MockIFT, mock_log):
        orch = MagicMock()
        orch.run_loop.return_value = {"stop_reason": "PASS", "history": []}
        q = _make_goal_queue(["go"])
        archive = MagicMock()
        run_goals_loop(_make_args(), q, orch, None, None, archive, Path("/tmp"))
        _, score = archive.record.call_args[0]
        assert score == 0.0

    def test_orchestrator_exception_handled_gracefully(self, MockTM, MockIFT, mock_log):
        orch = MagicMock()
        orch.run_loop.side_effect = RuntimeError("boom")
        q = _make_goal_queue(["broken goal"])
        archive = MagicMock()
        run_goals_loop(_make_args(), q, orch, None, None, archive, Path("/tmp"))
        archive.record.assert_called_once()

    def test_tracker_write_called_for_each_goal(self, MockTM, MockIFT, mock_log):
        orch = MagicMock()
        orch.run_loop.return_value = {"stop_reason": "PASS", "history": []}
        q = _make_goal_queue(["g1", "g2"])
        archive = MagicMock()
        tracker = MockIFT.return_value
        run_goals_loop(_make_args(), q, orch, None, None, archive, Path("/tmp"))
        assert tracker.write.call_count == 2

    def test_tracker_clear_called_even_on_exception(self, MockTM, MockIFT, mock_log):
        orch = MagicMock()
        orch.run_loop.side_effect = ValueError("err")
        q = _make_goal_queue(["bad goal"])
        archive = MagicMock()
        tracker = MockIFT.return_value
        run_goals_loop(_make_args(), q, orch, None, None, archive, Path("/tmp"))
        tracker.clear.assert_called_once()

    def test_external_goals_polled_when_queue_empty(self, MockTM, MockIFT, mock_log):
        orch = MagicMock()
        orch.poll_external_goals.return_value = []
        q = _make_goal_queue([])
        archive = MagicMock()
        run_goals_loop(_make_args(), q, orch, None, None, archive, Path("/tmp"))
        orch.poll_external_goals.assert_called()

    def test_poll_external_goals_exception_continues(self, MockTM, MockIFT, mock_log):
        orch = MagicMock()
        orch.poll_external_goals.side_effect = ConnectionError("network down")
        q = _make_goal_queue([])
        archive = MagicMock()
        # Should not raise
        run_goals_loop(_make_args(), q, orch, None, None, archive, Path("/tmp"))

    def test_polled_goals_added_to_queue(self, MockTM, MockIFT, mock_log):
        orch = MagicMock()
        orch.run_loop.return_value = {"stop_reason": "PASS", "history": []}

        added = []
        call_count = [0]

        def has_goals():
            call_count[0] += 1
            if call_count[0] == 1:
                return False  # triggers poll
            if call_count[0] == 2:
                return True  # goal was added
            return False

        orch.poll_external_goals.return_value = ["polled goal"]
        q = MagicMock()
        q.has_goals.side_effect = has_goals
        q.next.return_value = "polled goal"
        q.add.side_effect = lambda g: added.append(g)

        archive = MagicMock()
        run_goals_loop(_make_args(), q, orch, None, None, archive, Path("/tmp"))
        assert "polled goal" in added

    def test_multiple_goals_all_processed(self, MockTM, MockIFT, mock_log):
        orch = MagicMock()
        orch.run_loop.return_value = {"stop_reason": "PASS", "history": []}
        q = _make_goal_queue(["g1", "g2", "g3"])
        archive = MagicMock()
        run_goals_loop(_make_args(), q, orch, None, None, archive, Path("/tmp"))
        assert orch.run_loop.call_count == 3
        assert archive.record.call_count == 3

    def test_decompose_mode_calls_decompose_goal(self, MockTM, MockIFT, mock_log):
        orch = MagicMock()
        orch.run_loop.return_value = {"stop_reason": "PASS", "history": []}

        tm = MockTM.return_value
        subtask = MagicMock()
        subtask.status = "pending"
        subtask.title = "sub"
        root = MagicMock()
        root.subtasks = [subtask]
        tm.decompose_goal.return_value = root

        planner = MagicMock()
        q = _make_goal_queue(["big goal"])
        archive = MagicMock()
        run_goals_loop(_make_args(), q, orch, None, planner, archive, Path("/tmp"), decompose=True)
        tm.decompose_goal.assert_called_once_with("big goal", planner)

    def test_goal_processing_exception_does_not_crash_loop(self, MockTM, MockIFT, mock_log):
        orch = MagicMock()
        orch.run_loop.return_value = {"stop_reason": "PASS", "history": []}
        tracker = MockIFT.return_value
        tracker.write.side_effect = [OSError("disk full"), None]
        q = _make_goal_queue(["g1", "g2"])
        archive = MagicMock()
        # Should not raise; loop continues despite exception
        run_goals_loop(_make_args(), q, orch, None, None, archive, Path("/tmp"))
