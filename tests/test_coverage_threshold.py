"""
Tests for coverage threshold logic in core/quality_snapshot.py.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from core.quality_snapshot import check_coverage_thresholds

def test_check_coverage_thresholds_identifies_gaps():
    """
    Verify that the utility correctly identifies files below threshold.
    """
    mock_cov_data = {
        "files": {
            "core/orchestrator.py": {"summary": {"percent_covered": 85.0}},
            "agents/coder.py": {"summary": {"percent_covered": 10.0}},
            "utils/helper.py": {"summary": {"percent_covered": 0.0}}
        }
    }
    
    # Mocking json.loads(coverage.json) or similar source
    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.read_text", return_value='{"files": {}}'), \
         patch("json.loads", return_value=mock_cov_data):
        
        gaps = check_coverage_thresholds(
            project_root=Path("."),
            files=["core/orchestrator.py", "agents/coder.py", "utils/helper.py"],
            threshold=80.0
        )
        
        # Expected: coder.py and helper.py are below 80%
        assert len(gaps) == 2
        assert any(g["file"] == "agents/coder.py" for g in gaps)
        assert any(g["file"] == "utils/helper.py" for g in gaps)
        assert all(g["coverage"] < 80.0 for g in gaps)

def test_check_coverage_thresholds_returns_empty_when_all_pass():
    mock_cov_data = {
        "files": {
            "core/orchestrator.py": {"summary": {"percent_covered": 90.0}}
        }
    }
    
    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.read_text", return_value='{}'), \
         patch("json.loads", return_value=mock_cov_data):
        
        gaps = check_coverage_thresholds(
            project_root=Path("."),
            files=["core/orchestrator.py"],
            threshold=80.0
        )
        
        assert len(gaps) == 0
