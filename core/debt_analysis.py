from typing import List
import ast
import pytest
from dataclasses import dataclass

@dataclass
class DebtMetrics:
    coverage: float
    complexity: int 
    duplication: float
    untested_lines: List[int]
    flaky_rate: float

def analyze_test_coverage(test_file: str) -> DebtMetrics:
    """Analyze test file and return debt metrics"""
    metrics = DebtMetrics(
        coverage=0.0,
        complexity=0,
        duplication=0.0,
        untested_lines=[],
        flaky_rate=0.0
    )
    
    try:
        with open(test_file) as f:
            tree = ast.parse(f.read())
            
        # Calculate metrics
        metrics.complexity = count_cyclomatic_complexity(tree)
        metrics.duplication = detect_test_duplication(tree)
        metrics.coverage = calculate_test_coverage(test_file)
        metrics.untested_lines = find_untested_lines(test_file)
        metrics.flaky_rate = analyze_test_flakiness(test_file)
        
        return metrics
    except Exception as e:
        raise ValueError(f"Failed to analyze {test_file}: {str(e)}")

def count_cyclomatic_complexity(tree: ast.AST) -> int:
    """Calculate cyclomatic complexity of AST"""
    # Count branches in code
    complexity = 1
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
            complexity += 1
    return complexity

def detect_test_duplication(tree: ast.AST) -> float:
    """Calculate test code duplication percentage"""
    # Simple duplication detection
    lines = set()
    total_lines = 0
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith('test_'):
                func_lines = ast.unparse(node)
                lines.add(func_lines)
                total_lines += len(func_lines.split('\n'))
                
    if total_lines == 0:
        return 0.0
    return 1.0 - (len(lines) / total_lines)

def calculate_test_coverage(test_file: str) -> float:
    """Calculate test coverage percentage"""
    # Would integrate with coverage.py
    return 0.8 # Placeholder

def find_untested_lines(test_file: str) -> List[int]:
    """Find lines that lack test coverage"""
    # Would integrate with coverage.py
    return [42, 43] # Placeholder

def analyze_test_flakiness(test_file: str) -> float:
    """Analyze test flakiness rate"""
    # Would analyze test run history
    return 0.05 # Placeholder 

@pytest.fixture
def sample_test_file(tmp_path):
    test_content = """
import pytest
def test_something():
    assert True
def test_other():
    assert 1 == 1
"""
    p = tmp_path / "test_sample.py"
    p.write_text(test_content)
    return str(p)

def test_analyze_coverage(sample_test_file):
    metrics = analyze_test_coverage(sample_test_file)
    assert isinstance(metrics, DebtMetrics)
    assert 0 <= metrics.coverage <= 1.0
    assert metrics.complexity >= 0
    assert 0 <= metrics.duplication <= 1.0
    assert isinstance(metrics.untested_lines, list)
    assert 0 <= metrics.flaky_rate <= 1.0

def test_complexity_calculation():
    code = """
def test_func():
    if True:
        pass
    for x in range(5):
        if x > 2:
            pass
    """
    tree = ast.parse(code)
    complexity = count_cyclomatic_complexity(tree)
    assert complexity == 4  # Base + if + for + nested if

def test_duplication_detection():
    code = """
def test_a():
    x = 1
    assert x == 1
def test_b():
    x = 1
    assert x == 1
    """
    tree = ast.parse(code)
    duplication = detect_test_duplication(tree)
    assert duplication > 0.5  # High duplication expected
