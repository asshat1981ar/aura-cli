"""
Tests for StructuralAnalyzerSkill coverage reporting enhancement.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from agents.skills.structural_analyzer import StructuralAnalyzerSkill

@pytest.fixture
def analyzer():
    return StructuralAnalyzerSkill()

def test_structural_analyzer_reports_coverage_gaps(analyzer, tmp_path):
    """
    Verify that if 'report_coverage' is True, the skill identifies 
    files with 0% coverage as gaps.
    """
    project_root = tmp_path
    
    # Mock SymbolIndexerSkill to return a simple graph
    mock_indexer_res = {
        "import_graph": {
            "main.py": ["utils.py"],
            "utils.py": []
        }
    }
    
    # Mock ComplexityScorerSkill
    mock_complexity_res = {
        "file_avg_complexity": 5,
        "file_results": {
            "main.py": [{"name": "foo", "complexity": 10}],
            "utils.py": [{"name": "bar", "complexity": 5}]
        }
    }
    
    # Mock TestCoverageAnalyzerSkill
    # Let's say main.py has 100% and utils.py has 0%
    mock_coverage_res = {
        "coverage_pct": 50.0,
        "missing_files": ["utils.py"],
        "untested_functions": [{"file": "utils.py", "function": "bar"}],
        "method": "heuristic"
    }

    with patch("agents.skills.symbol_indexer.SymbolIndexerSkill.run", return_value=mock_indexer_res), \
         patch("agents.skills.complexity_scorer.ComplexityScorerSkill.run", return_value=mock_complexity_res), \
         patch("agents.skills.test_coverage_analyzer.TestCoverageAnalyzerSkill.run", return_value=mock_coverage_res):
        
        result = analyzer.run({
            "project_root": str(project_root),
            "report_coverage": True
        })
        
        assert "coverage_gaps" in result
        assert any(gap["file"] == "utils.py" for gap in result["coverage_gaps"])
        assert result["summary"].count("coverage gaps") > 0

def test_structural_analyzer_no_coverage_by_default(analyzer, tmp_path):
    """
    By default, it should NOT run coverage analysis (too slow).
    """
    project_root = tmp_path
    
    with patch("agents.skills.symbol_indexer.SymbolIndexerSkill.run", return_value={"import_graph": {}}), \
         patch("agents.skills.complexity_scorer.ComplexityScorerSkill.run", return_value={}), \
         patch("agents.skills.test_coverage_analyzer.TestCoverageAnalyzerSkill.run") as mock_cov:
        
        result = analyzer.run({
            "project_root": str(project_root)
        })
        
        assert "coverage_gaps" not in result
        mock_cov.assert_not_called()
