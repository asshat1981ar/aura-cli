import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from core.autonomous_discovery import AutonomousDiscovery

@pytest.fixture
def mock_goal_queue():
    return MagicMock()

@pytest.fixture
def mock_memory_store():
    return MagicMock()

@pytest.fixture
def autonomous_discovery(mock_goal_queue, mock_memory_store, tmp_path):
    return AutonomousDiscovery(mock_goal_queue, mock_memory_store, project_root=str(tmp_path))

@pytest.fixture(autouse=True)
def mock_structural_analyzer():
    with patch("agents.skills.structural_analyzer.StructuralAnalyzerSkill") as mock:
        mock_instance = mock.return_value
        mock_instance.run.return_value = {} # Default empty result
        yield mock

def test_initialization(autonomous_discovery):
    assert autonomous_discovery.queue is not None
    assert autonomous_discovery.memory is not None

def test_scan_no_files(autonomous_discovery):
    report = autonomous_discovery.run_scan()
    assert report["findings_total"] == 0
    assert report["new_goals"] == 0

def test_scan_with_todo(autonomous_discovery, tmp_path):
    # Create a dummy python file with a TODO
    d = tmp_path / "aura_cli"
    d.mkdir()
    p = d / "myfile.py"
    p.write_text("# TODO: Fix this thing\ndef foo(): pass")
    
    report = autonomous_discovery.run_scan()
    
    assert report["findings_total"] > 0
    assert report["new_goals"] > 0
    # verify goal content
    goals = report["goals"]
    assert any("TODO" in g for g in goals)
    assert any("Fix this thing" in g for g in goals)

def test_scan_skips_venv_todos(autonomous_discovery, tmp_path):
    venv_file = tmp_path / "venv" / "lib" / "python3.11" / "site-packages" / "pkg.py"
    venv_file.parent.mkdir(parents=True, exist_ok=True)
    venv_file.write_text("# TODO: ignore me\nx = 1")

    report = autonomous_discovery.run_scan()

    assert report["findings_total"] == 0
    assert report["new_goals"] == 0

def test_scan_missing_tests(autonomous_discovery, tmp_path):
    # Create a source file without a test
    src_dir = tmp_path / "core"
    src_dir.mkdir()
    (src_dir / "logic.py").write_text("def do_logic(): pass")
    
    # Create empty tests dir
    (tmp_path / "tests").mkdir()
    
    report = autonomous_discovery.run_scan()
    
    # Should find missing test for logic.py
    assert report["findings_total"] > 0
    goals = report["goals"]
    assert any("Add unit tests for core/logic.py" in g for g in goals)

def test_scan_structural_debt(autonomous_discovery, mock_structural_analyzer):
    mock_instance = mock_structural_analyzer.return_value
    mock_instance.run.return_value = {
        "circular_dependencies": [["a", "b", "a"]],
        "hotspots": [{"file": "f.py", "centrality": 0.9, "max_complexity": 10, "risk_level": "CRITICAL"}]
    }
    
    report = autonomous_discovery.run_scan()
    
    assert report["findings_total"] > 0
    goals = report["goals"]
    assert any("Fix circular dependency cycle" in g for g in goals)
    assert any("Refactor hotspot f.py" in g for g in goals)

def test_scan_long_function(autonomous_discovery, mock_structural_analyzer, tmp_path):
    # Create a file with a very long function (over 60 lines)
    d = tmp_path / "core"
    d.mkdir()
    p = d / "long_func.py"
    lines = ["def huge_func():"] + ["    x = 1"] * 70
    p.write_text("\n".join(lines))
    
    report = autonomous_discovery.run_scan()
    
    assert report["findings_total"] > 0
    goals = report["goals"]
    # Check if any goal mentions refactoring long function
    assert any("Refactor long function" in g for g in goals)
    assert any("long_func.py" in g for g in goals)
