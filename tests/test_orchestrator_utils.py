"""Tests for core/orchestrator_utils.py."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from core.orchestrator_utils import (
    analyze_error,
    generate_cycle_id,
    load_json_config,
    snapshot_file_state,
    restore_file_snapshots,
    normalize_verification_result,
    route_verification_failure,
    calculate_retry_delay,
    format_execution_summary,
    merge_skill_context,
    should_retry_phase,
    extract_code_from_response,
    sanitize_goal_for_logging,
    BeadsSyncLoop,
)


class TestBeadsSyncLoop:
    """Test BeadsSyncLoop class."""
    
    def test_init(self):
        """Test initialization."""
        skill = Mock()
        loop = BeadsSyncLoop(skill)
        assert loop._skill is skill
        assert loop._n == 0
    
    def test_on_cycle_complete_dry_run(self):
        """Test that dry_run cycles don't trigger sync."""
        skill = Mock()
        loop = BeadsSyncLoop(skill)
        
        loop.on_cycle_complete({"dry_run": True})
        assert loop._n == 0
        skill.run.assert_not_called()
    
    def test_on_cycle_complete_triggers_every_n(self):
        """Test sync triggers every N cycles."""
        skill = Mock()
        loop = BeadsSyncLoop(skill)
        
        # Trigger 4 cycles (should not sync yet)
        for _ in range(4):
            loop.on_cycle_complete({})
        
        assert loop._n == 4
        skill.run.assert_not_called()
        
        # 5th cycle triggers sync
        loop.on_cycle_complete({})
        assert loop._n == 5
        assert skill.run.call_count == 2  # pull and push


class TestAnalyzeError:
    """Test analyze_error function."""
    
    def test_syntax_error(self):
        """Test syntax error detection."""
        assert analyze_error("SyntaxError: invalid syntax") == "syntax_fix"
        assert analyze_error("IndentationError") == "syntax_fix"
    
    def test_import_error(self):
        """Test import error detection."""
        assert analyze_error("ImportError: No module named 'foo'") == "dependency_check"
        assert analyze_error("ModuleNotFoundError") == "dependency_check"
    
    def test_permission_error(self):
        """Test permission error detection."""
        assert analyze_error("Permission denied") == "permission_fix"
        assert analyze_error("Access denied") == "permission_fix"
    
    def test_timeout_error(self):
        """Test timeout error detection."""
        assert analyze_error("Timeout error") == "retry_with_timeout"
        assert analyze_error("Deadline exceeded") == "retry_with_timeout"
    
    def test_memory_error(self):
        """Test memory error detection."""
        assert analyze_error("MemoryError") == "memory_optimization"
    
    def test_unknown_error(self):
        """Test unknown error returns None."""
        assert analyze_error("Some random error") is None


class TestGenerateCycleId:
    """Test generate_cycle_id function."""
    
    def test_format(self):
        """Test cycle ID format."""
        cycle_id = generate_cycle_id()
        assert cycle_id.startswith("cycle_")
        assert len(cycle_id) > 20  # prefix + uuid + timestamp
    
    def test_uniqueness(self):
        """Test cycle IDs are unique."""
        ids = [generate_cycle_id() for _ in range(10)]
        assert len(set(ids)) == 10


class TestLoadJsonConfig:
    """Test load_json_config function."""
    
    def test_load_valid_config(self):
        """Test loading valid JSON config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"key": "value", "number": 42}, f)
            f.flush()
            
            result = load_json_config(Path(f.name))
            assert result == {"key": "value", "number": 42}
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file returns empty dict."""
        result = load_json_config(Path("/nonexistent/path.json"))
        assert result == {}
    
    def test_load_invalid_json(self):
        """Test loading invalid JSON returns empty dict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            f.flush()
            
            result = load_json_config(Path(f.name))
            assert result == {}


class TestSnapshotFileState:
    """Test snapshot_file_state function."""
    
    def test_snapshot_existing_file(self):
        """Test snapshot of existing file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("print('hello')")
            f.flush()
            
            snapshot = snapshot_file_state(f.name)
            
            assert snapshot["path"] == f.name
            assert snapshot["existed"] is True
            assert snapshot["content"] == "print('hello')"
            assert "timestamp" in snapshot
    
    def test_snapshot_nonexistent_file(self):
        """Test snapshot of non-existent file."""
        snapshot = snapshot_file_state("/nonexistent/file.py")
        
        assert snapshot["path"] == "/nonexistent/file.py"
        assert snapshot["existed"] is False
        assert snapshot["content"] is None


class TestRestoreFileSnapshots:
    """Test restore_file_snapshots function."""
    
    def test_restore_existing_file(self):
        """Test restoring existing file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("original content")
            f.flush()
            
            # Modify file
            Path(f.name).write_text("modified content")
            
            # Restore
            snapshot = {
                "path": f.name,
                "existed": True,
                "content": "original content",
            }
            restore_file_snapshots([snapshot])
            
            assert Path(f.name).read_text() == "original content"
    
    def test_restore_delete_created_file(self):
        """Test deleting file that didn't exist before."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_file = Path(tmpdir) / "new_file.py"
            new_file.write_text("new content")
            
            snapshot = {
                "path": str(new_file),
                "existed": False,
                "content": None,
            }
            restore_file_snapshots([snapshot])
            
            assert not new_file.exists()


class TestNormalizeVerificationResult:
    """Test normalize_verification_result function."""
    
    def test_normalize_with_status_pass(self):
        """Test normalizing result with pass status."""
        result = normalize_verification_result({"status": "pass"})
        assert result["success"] is True
    
    def test_normalize_with_status_fail(self):
        """Test normalizing result with fail status."""
        result = normalize_verification_result({"status": "fail"})
        assert result["success"] is False
    
    def test_normalize_with_success_field(self):
        """Test normalizing result with success field."""
        result = normalize_verification_result({"success": True})
        assert result["success"] is True
    
    def test_normalize_invalid_format(self):
        """Test normalizing invalid format."""
        result = normalize_verification_result("invalid")
        assert result["success"] is False
        assert "Invalid verification result format" in result["errors"]


class TestRouteVerificationFailure:
    """Test route_verification_failure function."""
    
    def test_syntax_error_routes_to_retry(self):
        """Test syntax errors route to retry."""
        result = route_verification_failure({"errors": ["SyntaxError"] })
        assert result == "retry"
    
    def test_test_failure_routes_to_replan(self):
        """Test test failures route to replan."""
        result = route_verification_failure({"failed": 1, "errors": []})
        assert result == "replan"
    
    def test_sandbox_error_routes_to_escalate(self):
        """Test sandbox errors route to escalate."""
        result = route_verification_failure({"errors": ["sandbox execution failed"]})
        assert result == "escalate"
    
    def test_timeout_routes_to_retry(self):
        """Test timeout routes to retry."""
        result = route_verification_failure({"errors": ["timeout exceeded"]})
        assert result == "retry"
    
    def test_default_routes_to_retry(self):
        """Test default routes to retry."""
        result = route_verification_failure({"errors": []})
        assert result == "retry"


class TestCalculateRetryDelay:
    """Test calculate_retry_delay function."""
    
    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        assert calculate_retry_delay(0) == 1.0
        assert calculate_retry_delay(1) == 2.0
        assert calculate_retry_delay(2) == 4.0
        assert calculate_retry_delay(3) == 8.0
    
    def test_max_delay_cap(self):
        """Test maximum delay cap."""
        delay = calculate_retry_delay(10, max_delay=30.0)
        assert delay <= 30.0


class TestFormatExecutionSummary:
    """Test format_execution_summary function."""
    
    def test_summary_format(self):
        """Test summary format."""
        summary = format_execution_summary(
            goal="Test goal",
            cycle_id="cycle_123",
            stop_reason="completed",
            cycles=3,
            phase_outputs={"plan": {}, "act": {}},
            duration_seconds=10.5,
        )
        
        assert summary["goal"] == "Test goal"
        assert summary["cycle_id"] == "cycle_123"
        assert summary["stop_reason"] == "completed"
        assert summary["cycles"] == 3
        assert summary["duration_seconds"] == 10.5
        assert summary["phases_completed"] == ["plan", "act"]
        assert "timestamp" in summary
    
    def test_long_goal_truncation(self):
        """Test long goal truncation."""
        long_goal = "x" * 300
        summary = format_execution_summary(
            goal=long_goal,
            cycle_id="cycle_123",
            stop_reason="completed",
            cycles=1,
            phase_outputs={},
            duration_seconds=1.0,
        )
        
        assert len(summary["goal"]) <= 200


class TestMergeSkillContext:
    """Test merge_skill_context function."""
    
    def test_merge_empty_skill_results(self):
        """Test merging empty skill results."""
        base = {"key": "value"}
        result = merge_skill_context(base, {})
        assert result == base
    
    def test_merge_with_issues(self):
        """Test merging with issues."""
        base = {"key": "value"}
        skills = {"issues": ["issue1"], "suggestions": ["suggestion1"]}
        
        result = merge_skill_context(base, skills)
        
        assert result["key"] == "value"
        assert result["skill_analysis"] == skills
        assert result["known_issues"] == ["issue1"]
        assert result["implementation_hints"] == ["suggestion1"]


class TestShouldRetryPhase:
    """Test should_retry_phase function."""
    
    def test_under_max_retries(self):
        """Test retry when under max."""
        assert should_retry_phase(0, 3, "syntax_error") is True
        assert should_retry_phase(2, 3, "syntax_error") is True
    
    def test_at_max_retries(self):
        """Test no retry at max."""
        assert should_retry_phase(3, 3, "syntax_error") is False
    
    def test_non_retryable_errors(self):
        """Test non-retryable errors."""
        assert should_retry_phase(0, 3, "permission_denied") is False
        assert should_retry_phase(0, 3, "auth_error") is False


class TestExtractCodeFromResponse:
    """Test extract_code_from_response function."""
    
    def test_extract_from_code_block(self):
        """Test extracting from markdown code block."""
        response = """Here's the code:
```python
def foo():
    return 42
```
"""
        code = extract_code_from_response(response)
        assert "def foo():" in code
    
    def test_extract_from_plain_string(self):
        """Test extracting from plain string."""
        response = "def bar(): pass"
        code = extract_code_from_response(response)
        assert code == "def bar(): pass"
    
    def test_extract_from_dict(self):
        """Test extracting from dict response."""
        response = {"code": "def baz(): pass"}
        code = extract_code_from_response(response)
        assert code == "def baz(): pass"
    
    def test_extract_from_dict_alternative_keys(self):
        """Test extracting from dict with alternative keys."""
        assert extract_code_from_response({"implementation": "impl"}) == "impl"
        assert extract_code_from_response({"content": "content"}) == "content"


class TestSanitizeGoalForLogging:
    """Test sanitize_goal_for_logging function."""
    
    def test_short_goal_unchanged(self):
        """Test short goal is unchanged."""
        goal = "Short goal"
        assert sanitize_goal_for_logging(goal) == "Short goal"
    
    def test_newlines_removed(self):
        """Test newlines are removed."""
        goal = "Line 1\nLine 2\rLine 3"
        assert "\n" not in sanitize_goal_for_logging(goal)
        assert "\r" not in sanitize_goal_for_logging(goal)
    
    def test_long_goal_truncated(self):
        """Test long goal is truncated."""
        goal = "x" * 300
        sanitized = sanitize_goal_for_logging(goal, max_length=100)
        assert len(sanitized) <= 100
        assert sanitized.endswith("...")
