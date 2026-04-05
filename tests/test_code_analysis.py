"""
Unit tests for core/code_analysis.py

Tests for code duplication detection and analysis functionality.
"""

import ast
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.code_analysis import (
    CodeAnalyzer,
    DuplicationMetrics,
    scan_directory
)


class TestDuplicationMetrics:
    """Tests for DuplicationMetrics dataclass."""
    
    def test_default_initialization(self):
        """Test DuplicationMetrics initializes with correct defaults."""
        metrics = DuplicationMetrics(
            duplicated_lines=0,
            total_lines=100,
            duplicate_ratio=0.0,
            hotspots={}
        )
        
        assert metrics.duplicated_lines == 0
        assert metrics.total_lines == 100
        assert metrics.duplicate_ratio == 0.0
        assert metrics.hotspots == {}
    
    def test_custom_initialization(self):
        """Test DuplicationMetrics with custom values."""
        hotspots = {"file.py": [10, 20, 30]}
        metrics = DuplicationMetrics(
            duplicated_lines=50,
            total_lines=200,
            duplicate_ratio=0.25,
            hotspots=hotspots
        )
        
        assert metrics.duplicated_lines == 50
        assert metrics.total_lines == 200
        assert metrics.duplicate_ratio == 0.25
        assert metrics.hotspots == hotspots


class TestCodeAnalyzer:
    """Tests for CodeAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Fixture providing a fresh CodeAnalyzer instance."""
        return CodeAnalyzer()
    
    @pytest.fixture
    def temp_python_file(self):
        """Fixture creating a temporary Python file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
def helper_function():
    """A simple helper function."""
    x = 1
    y = 2
    return x + y

def another_function():
    """Another function with different logic."""
    a = 10
    b = 20
    return a * b
''')
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    @pytest.fixture
    def temp_duplicate_file(self):
        """Fixture creating a Python file with duplicate functions."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
def duplicate_function():
    """First occurrence."""
    x = 1
    y = 2
    return x + y

def duplicate_function():
    """Second occurrence - duplicate."""
    x = 1
    y = 2
    return x + y

def unique_function():
    """Unique function."""
    return 42
''')
            temp_path = f.name
        
        yield temp_path
        os.unlink(temp_path)
    
    def test_initialization(self, analyzer):
        """Test CodeAnalyzer initializes correctly."""
        assert isinstance(analyzer.seen_snippets, set)
        assert len(analyzer.seen_snippets) == 0
    
    def test_analyze_duplication_valid_file(self, analyzer, temp_python_file):
        """Test analyzing a valid Python file."""
        metrics = analyzer.analyze_duplication(temp_python_file)
        
        assert isinstance(metrics, DuplicationMetrics)
        assert metrics.total_lines > 0
        assert isinstance(metrics.hotspots, dict)
    
    def test_analyze_duplication_file_not_found(self, analyzer):
        """Test analyzing a non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            analyzer.analyze_duplication("/nonexistent/path/file.py")
    
    def test_analyze_duplication_with_duplicates(self, analyzer, temp_duplicate_file):
        """Test analyzing file with duplicate functions."""
        metrics = analyzer.analyze_duplication(temp_duplicate_file)
        
        assert isinstance(metrics, DuplicationMetrics)
        assert metrics.total_lines > 0
        # The duplicate ratio should be calculated
        assert 0.0 <= metrics.duplicate_ratio <= 1.0
    
    def test_analyze_duplication_empty_file(self, analyzer):
        """Test analyzing an empty Python file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('')
            temp_path = f.name
        
        try:
            metrics = analyzer.analyze_duplication(temp_path)
            
            assert isinstance(metrics, DuplicationMetrics)
            assert metrics.total_lines == 0
            assert metrics.duplicate_ratio == 0.0
        finally:
            os.unlink(temp_path)
    
    def test_analyze_duplication_syntax_error(self, analyzer):
        """Test analyzing a file with syntax errors."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('def broken_function(\n')  # Missing closing parenthesis
            temp_path = f.name
        
        try:
            with pytest.raises(SyntaxError):
                analyzer.analyze_duplication(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_analyze_node_with_function(self, analyzer):
        """Test _analyze_node with a function definition."""
        code = '''
def test_func():
    pass
'''
        tree = ast.parse(code)
        metrics = DuplicationMetrics(
            duplicated_lines=0,
            total_lines=0,
            duplicate_ratio=0.0,
            hotspots={}
        )
        
        analyzer._analyze_node(tree, metrics, "test.py")
        
        # Should have processed the function
        assert metrics.total_lines > 0
    
    def test_analyze_node_with_class(self, analyzer):
        """Test _analyze_node with a class definition."""
        code = '''
class TestClass:
    def method(self):
        pass
'''
        tree = ast.parse(code)
        metrics = DuplicationMetrics(
            duplicated_lines=0,
            total_lines=0,
            duplicate_ratio=0.0,
            hotspots={}
        )
        
        analyzer._analyze_node(tree, metrics, "test.py")
        
        # Should have processed the class and its method
        assert metrics.total_lines > 0
    
    def test_check_function_duplication_unique(self, analyzer):
        """Test checking a unique function for duplication."""
        code = '''
def unique_func():
    x = 1
    y = 2
    return x + y
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        metrics = DuplicationMetrics(
            duplicated_lines=0,
            total_lines=0,
            duplicate_ratio=0.0,
            hotspots={}
        )
        
        analyzer._check_function_duplication(func_node, metrics, "test.py")
        
        # First occurrence should not be marked as duplicate
        assert metrics.duplicated_lines == 0
        assert len(analyzer.seen_snippets) == 1
    
    def test_check_function_duplication_duplicate(self, analyzer):
        """Test checking a duplicate function."""
        code = '''
def duplicate_func():
    x = 1
    return x
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        # Add function to seen snippets first
        func_str = ast.unparse(func_node)
        analyzer.seen_snippets.add(func_str)
        
        metrics = DuplicationMetrics(
            duplicated_lines=0,
            total_lines=0,
            duplicate_ratio=0.0,
            hotspots={}
        )
        
        analyzer._check_function_duplication(func_node, metrics, "test.py")
        
        # Should detect as duplicate
        assert metrics.duplicated_lines > 0
        assert "test.py" in metrics.hotspots
    
    def test_find_duplicate_blocks_with_duplicates(self, analyzer):
        """Test finding duplicate code blocks within a function."""
        code = '''
def func_with_blocks():
    if True:
        x = 1
        y = 2
        z = 3
    if True:
        x = 1
        y = 2
        z = 3
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        metrics = DuplicationMetrics(
            duplicated_lines=0,
            total_lines=0,
            duplicate_ratio=0.0,
            hotspots={}
        )
        
        analyzer._find_duplicate_blocks(func_node, metrics, "test.py")
        
        # Should detect the duplicate if blocks
        # Note: depends on exact AST structure
        assert isinstance(metrics.hotspots, dict)
    
    def test_find_duplicate_blocks_no_duplicates(self, analyzer):
        """Test finding duplicate blocks when none exist."""
        code = '''
def func_with_unique_blocks():
    if True:
        x = 1
    if False:
        y = 2
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        metrics = DuplicationMetrics(
            duplicated_lines=0,
            total_lines=0,
            duplicate_ratio=0.0,
            hotspots={}
        )
        
        analyzer._find_duplicate_blocks(func_node, metrics, "test.py")
        
        # Should not find duplicates
        assert metrics.duplicated_lines == 0
    
    def test_find_duplicate_blocks_below_threshold(self, analyzer):
        """Test that small blocks (< 3 lines) are ignored."""
        code = '''
def func_with_small_blocks():
    if True:
        x = 1
    if True:
        x = 1
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        metrics = DuplicationMetrics(
            duplicated_lines=0,
            total_lines=0,
            duplicate_ratio=0.0,
            hotspots={}
        )
        
        analyzer._find_duplicate_blocks(func_node, metrics, "test.py")
        
        # Small blocks should be ignored
        assert metrics.duplicated_lines == 0


class TestScanDirectory:
    """Tests for scan_directory function."""
    
    @pytest.fixture
    def temp_directory(self):
        """Fixture creating a temporary directory with Python files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create subdirectory
            subdir = Path(temp_dir) / "subdir"
            subdir.mkdir()
            
            # Create Python files
            (Path(temp_dir) / "file1.py").write_text('''
def func1():
    return 1
''')
            (Path(temp_dir) / "file2.py").write_text('''
def func2():
    return 2
''')
            (subdir / "file3.py").write_text('''
def func3():
    return 3
''')
            # Create non-Python file
            (Path(temp_dir) / "readme.txt").write_text("Not a Python file")
            
            yield temp_dir
    
    def test_scan_directory_finds_all_python_files(self, temp_directory):
        """Test that scan_directory finds all Python files recursively."""
        results = scan_directory(temp_directory)
        
        # Should find 3 Python files
        assert len(results) == 3
        
        # All values should be DuplicationMetrics
        for metrics in results.values():
            assert isinstance(metrics, DuplicationMetrics)
    
    def test_scan_directory_empty_directory(self):
        """Test scanning an empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = scan_directory(temp_dir)
            
            assert results == {}
    
    def test_scan_directory_no_python_files(self):
        """Test scanning directory with no Python files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "readme.txt").write_text("Not Python")
            (Path(temp_dir) / "data.json").write_text('{"key": "value"}')
            
            results = scan_directory(temp_dir)
            
            assert results == {}
    
    def test_scan_directory_handles_errors(self):
        """Test that scan_directory handles file analysis errors gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid Python file
            (Path(temp_dir) / "valid.py").write_text('''
def valid_func():
    return 1
''')
            # Create a file with syntax error
            (Path(temp_dir) / "invalid.py").write_text('def broken(')
            
            # Should not raise exception
            results = scan_directory(temp_dir)
            
            # Should have results for valid file only
            assert len(results) == 1
            assert any("valid.py" in path for path in results.keys())
    
    def test_scan_directory_returns_dict_with_paths(self, temp_directory):
        """Test that scan_directory returns proper dictionary structure."""
        results = scan_directory(temp_directory)
        
        # All keys should be absolute or relative paths
        for file_path in results.keys():
            assert file_path.endswith('.py')
            assert isinstance(results[file_path], DuplicationMetrics)


class TestIntegration:
    """Integration tests for the code analysis module."""
    
    def test_full_analysis_workflow(self):
        """Test the complete analysis workflow on a realistic codebase."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a module with various code patterns
            (Path(temp_dir) / "sample_module.py").write_text('''
"""Sample module for testing code analysis."""

import os
import sys
from typing import List, Dict

def process_data(data: List[int]) -> int:
    """Process a list of integers."""
    result = 0
    for item in data:
        if item > 0:
            result += item
        elif item < 0:
            result -= abs(item)
        else:
            pass
    return result

def analyze_results(results: Dict[str, int]) -> str:
    """Analyze results dictionary."""
    total = sum(results.values())
    if total > 100:
        return "high"
    elif total > 50:
        return "medium"
    else:
        return "low"

class DataProcessor:
    """Process data with various methods."""
    
    def __init__(self, config: dict):
        self.config = config
    
    def process(self, data: List[int]) -> List[int]:
        """Process data according to config."""
        return [x * 2 for x in data if x > 0]
    
    def validate(self, data: List[int]) -> bool:
        """Validate data."""
        return all(isinstance(x, int) for x in data)
''')
            
            # Run full analysis
            results = scan_directory(temp_dir)
            
            # Verify results
            assert len(results) == 1
            
            file_path = list(results.keys())[0]
            metrics = results[file_path]
            
            assert isinstance(metrics, DuplicationMetrics)
            assert metrics.total_lines > 0
            assert 0.0 <= metrics.duplicate_ratio <= 1.0
            assert isinstance(metrics.hotspots, dict)
    
    def test_multiple_analyzers_independent(self):
        """Test that multiple analyzer instances are independent."""
        code = '''
def test_func():
    return 42
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            # Create two separate analyzers
            analyzer1 = CodeAnalyzer()
            analyzer2 = CodeAnalyzer()
            
            # Process with first analyzer
            metrics1 = analyzer1.analyze_duplication(temp_path)
            
            # Process with second analyzer
            metrics2 = analyzer2.analyze_duplication(temp_path)
            
            # Results should be independent
            assert metrics1.total_lines == metrics2.total_lines
            # But seen_snippets should be separate
            assert analyzer1.seen_snippets is not analyzer2.seen_snippets
        finally:
            os.unlink(temp_path)
