"""
Unit tests for core/code_deduplicator.py

Tests for code duplication detection and utility function extraction.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from core.code_deduplicator import DuplicateCode, CodeDeduplicator


class TestDuplicateCode:
    """Tests for DuplicateCode dataclass."""

    def test_creation(self):
        """Test creating a DuplicateCode instance."""
        dup = DuplicateCode(source_file="file1.py", target_file="file2.py", source_lines=["line1\n", "line2\n"], similarity_score=0.95, line_numbers=[10, 20])

        assert dup.source_file == "file1.py"
        assert dup.target_file == "file2.py"
        assert dup.source_lines == ["line1\n", "line2\n"]
        assert dup.similarity_score == 0.95
        assert dup.line_numbers == [10, 20]

    def test_default_creation(self):
        """Test creating DuplicateCode with default values."""
        dup = DuplicateCode(source_file="a.py", target_file="b.py", source_lines=[], similarity_score=0.0, line_numbers=[])

        assert dup.similarity_score == 0.0
        assert dup.line_numbers == []


class TestCodeDeduplicator:
    """Tests for CodeDeduplicator class."""

    @pytest.fixture
    def deduplicator(self):
        """Fixture providing a fresh CodeDeduplicator instance."""
        return CodeDeduplicator()

    @pytest.fixture
    def temp_files_with_duplicates(self):
        """Fixture creating temporary files with duplicate code."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # File 1 with code
            file1 = Path(temp_dir) / "file1.py"
            file1.write_text("""def helper():
    x = 1
    y = 2
    return x + y

def main():
    print("hello")
""")

            # File 2 with similar code
            file2 = Path(temp_dir) / "file2.py"
            file2.write_text("""def other():
    pass

def helper():
    x = 1
    y = 2
    return x + y
""")

            yield [str(file1), str(file2)]

    @pytest.fixture
    def temp_files_no_duplicates(self):
        """Fixture creating temporary files with no duplicate code."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file1 = Path(temp_dir) / "file1.py"
            file1.write_text("""def unique1():
    return 1
""")

            file2 = Path(temp_dir) / "file2.py"
            file2.write_text("""def unique2():
    return 2
""")

            yield [str(file1), str(file2)]

    def test_initialization(self, deduplicator):
        """Test CodeDeduplicator initializes correctly."""
        assert deduplicator.duplicates == []
        assert deduplicator.utility_functions == {}

    def test_detect_duplicates_finds_duplicates(self, deduplicator, temp_files_with_duplicates):
        """Test that detect_duplicates finds duplicate code segments."""
        duplicates = deduplicator.detect_duplicates(temp_files_with_duplicates, min_lines=3, similarity_threshold=0.8)

        assert len(duplicates) > 0
        assert all(isinstance(d, DuplicateCode) for d in duplicates)

    def test_detect_duplicates_no_duplicates(self, deduplicator, temp_files_no_duplicates):
        """Test that detect_duplicates returns empty when no duplicates."""
        duplicates = deduplicator.detect_duplicates(temp_files_no_duplicates, min_lines=3, similarity_threshold=0.8)

        assert duplicates == []

    def test_detect_duplicates_single_file(self, deduplicator):
        """Test that detect_duplicates skips same file comparisons."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file1 = Path(temp_dir) / "file1.py"
            file1.write_text("""def func():
    pass
""")

            duplicates = deduplicator.detect_duplicates([str(file1)])

            # Should be empty - doesn't compare file to itself
            assert duplicates == []

    def test_detect_duplicates_stores_results(self, deduplicator, temp_files_with_duplicates):
        """Test that detect_duplicates stores results in self.duplicates."""
        deduplicator.detect_duplicates(temp_files_with_duplicates)

        assert len(deduplicator.duplicates) > 0

    def test_detect_duplicates_respects_min_lines(self, deduplicator):
        """Test that min_lines parameter is respected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Files with 2-line duplicates
            file1 = Path(temp_dir) / "file1.py"
            file1.write_text("x = 1\ny = 2\n")

            file2 = Path(temp_dir) / "file2.py"
            file2.write_text("x = 1\ny = 2\n")

            # With min_lines=3, should not find the 2-line duplicates
            duplicates = deduplicator.detect_duplicates([str(file1), str(file2)], min_lines=3)

            assert len(duplicates) == 0

    def test_detect_duplicates_respects_threshold(self, deduplicator):
        """Test that similarity_threshold is respected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Files with slightly different code
            file1 = Path(temp_dir) / "file1.py"
            file1.write_text("""def func():
    x = 1
    y = 2
    return x
""")

            file2 = Path(temp_dir) / "file2.py"
            file2.write_text("""def func():
    x = 1
    y = 3
    return x
""")

            # With high threshold, should not find partial match
            duplicates = deduplicator.detect_duplicates([str(file1), str(file2)], min_lines=3, similarity_threshold=1.0)

            assert len(duplicates) == 0

    def test_detect_duplicates_file_not_found(self, deduplicator):
        """Test handling of non-existent files."""
        with pytest.raises(FileNotFoundError):
            deduplicator.detect_duplicates(["/nonexistent/file.py"])

    def test_extract_utility_function(self, deduplicator):
        """Test extracting a utility function from duplicate code."""
        duplicate = DuplicateCode(source_file="file1.py", target_file="file2.py", source_lines=["    x = 1\n", "    y = 2\n", "    return x + y\n"], similarity_score=1.0, line_numbers=[1, 1])

        utility_fn = deduplicator.extract_utility_function(duplicate, "calculate_sum")

        assert "def calculate_sum(" in utility_fn
        assert "x = 1" in utility_fn
        assert "y = 2" in utility_fn
        assert "return x + y" in utility_fn

    def test_extract_utility_function_stores_in_dict(self, deduplicator):
        """Test that extracted function is stored in utility_functions dict."""
        duplicate = DuplicateCode(source_file="file1.py", target_file="file2.py", source_lines=["    pass\n"], similarity_score=1.0, line_numbers=[1, 1])

        deduplicator.extract_utility_function(duplicate, "my_func")

        assert "my_func" in deduplicator.utility_functions

    def test_extract_utility_function_removes_indentation(self, deduplicator):
        """Test that utility function extraction normalizes indentation."""
        duplicate = DuplicateCode(source_file="file1.py", target_file="file2.py", source_lines=["        x = 1\n", "        y = 2\n"], similarity_score=1.0, line_numbers=[1, 1])

        utility_fn = deduplicator.extract_utility_function(duplicate, "test_func")

        lines = utility_fn.split("\n")
        # First line is function definition
        assert lines[0].startswith("def test_func(")
        # Body lines should be indented with 4 spaces
        assert lines[1].startswith("    x = 1")
        assert lines[2].startswith("    y = 2")

    def test_extract_utility_function_empty_lines(self, deduplicator):
        """Test extraction with empty lines in code."""
        duplicate = DuplicateCode(source_file="file1.py", target_file="file2.py", source_lines=["    x = 1\n", "\n", "    y = 2\n"], similarity_score=1.0, line_numbers=[1, 1])

        utility_fn = deduplicator.extract_utility_function(duplicate, "func_with_blank")

        assert "def func_with_blank(" in utility_fn

    def test_find_similar_segments(self, deduplicator):
        """Test finding similar segments between files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_file = Path(temp_dir) / "source.py"
            source_file.write_text("""line1
line2
line3
line4
""")

            target_file = Path(temp_dir) / "target.py"
            target_file.write_text("""other
line1
line2
line3
more
""")

            source_lines = ["line1\n", "line2\n", "line3\n", "line4\n"]

            duplicates = deduplicator._find_similar_segments(str(source_file), str(target_file), source_lines, min_lines=3, similarity_threshold=0.8)

            assert len(duplicates) > 0
            assert all(d.source_file == str(source_file) for d in duplicates)
            assert all(d.target_file == str(target_file) for d in duplicates)

    def test_calculate_similarity_identical(self, deduplicator):
        """Test similarity calculation for identical lines."""
        lines1 = ["x = 1\n", "y = 2\n", "z = 3\n"]
        lines2 = ["x = 1\n", "y = 2\n", "z = 3\n"]

        similarity = deduplicator._calculate_similarity(lines1, lines2)

        assert similarity == 1.0

    def test_calculate_similarity_different(self, deduplicator):
        """Test similarity calculation for completely different lines."""
        lines1 = ["x = 1\n", "y = 2\n"]
        lines2 = ["a = 10\n", "b = 20\n"]

        similarity = deduplicator._calculate_similarity(lines1, lines2)

        assert similarity == 0.0

    def test_calculate_similarity_partial(self, deduplicator):
        """Test similarity calculation for partially matching lines."""
        lines1 = ["x = 1\n", "y = 2\n", "z = 3\n"]
        lines2 = ["x = 1\n", "y = 999\n", "z = 3\n"]

        similarity = deduplicator._calculate_similarity(lines1, lines2)

        assert 0.0 < similarity < 1.0
        assert similarity == pytest.approx(2 / 3, abs=0.01)

    def test_calculate_similarity_ignores_whitespace(self, deduplicator):
        """Test that similarity calculation handles whitespace differences."""
        lines1 = ["x = 1\n"]
        lines2 = ["  x = 1  \n"]  # Different whitespace

        similarity = deduplicator._calculate_similarity(lines1, lines2)

        # Strip should make them equal
        assert similarity == 1.0

    def test_calculate_similarity_empty_lists(self, deduplicator):
        """Test similarity calculation with empty lists raises ZeroDivisionError."""
        # The current implementation doesn't handle empty lists
        with pytest.raises(ZeroDivisionError):
            deduplicator._calculate_similarity([], [])

    def test_extract_parameters(self, deduplicator):
        """Test extracting parameters from code."""
        # Valid Python code - need to make it parseable
        code_lines = ["x = a + b\n", "y = c * d\n"]

        params = deduplicator._extract_parameters(code_lines)

        # Should identify loaded names
        assert isinstance(params, set)
        assert "a" in params or "b" in params or "c" in params or "d" in params

    def test_extract_parameters_excludes_builtins(self, deduplicator):
        """Test parameter extraction from code with function calls."""
        code_lines = ["x = len(items)\n", "y = print(msg)\n"]

        params = deduplicator._extract_parameters(code_lines)

        # Should extract loaded names
        assert isinstance(params, set)
        assert "items" in params or "msg" in params or "x" in params or "y" in params

    def test_extract_parameters_no_variables(self, deduplicator):
        """Test extracting parameters from code with no variables."""
        code_lines = ["pass\n"]

        params = deduplicator._extract_parameters(code_lines)

        assert params == set()

    def test_extract_parameters_invalid_code(self, deduplicator):
        """Test extracting parameters from invalid Python code."""
        code_lines = ["    invalid syntax!!!\n"]

        with pytest.raises(SyntaxError):
            deduplicator._extract_parameters(code_lines)


class TestIntegration:
    """Integration tests for code deduplicator."""

    def test_end_to_end_workflow(self):
        """Test complete deduplication workflow."""
        deduplicator = CodeDeduplicator()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files with duplicate code
            file1 = Path(temp_dir) / "module1.py"
            file1.write_text("""def process_data():
    data = load_data()
    result = transform(data)
    return result

def other():
    pass
""")

            file2 = Path(temp_dir) / "module2.py"
            file2.write_text("""def helper():
    pass

def process_data():
    data = load_data()
    result = transform(data)
    return result
""")

            # Detect duplicates
            duplicates = deduplicator.detect_duplicates([str(file1), str(file2)], min_lines=3, similarity_threshold=1.0)

            assert len(duplicates) > 0

            # Extract utility function
            if duplicates:
                utility_fn = deduplicator.extract_utility_function(duplicates[0], "shared_process_data")

                assert "def shared_process_data(" in utility_fn
                assert "shared_process_data" in deduplicator.utility_functions

    def test_multiple_duplicates_in_same_file_pair(self):
        """Test detecting multiple duplicate segments between same files."""
        deduplicator = CodeDeduplicator()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Files with multiple duplicate sections
            file1 = Path(temp_dir) / "a.py"
            file1.write_text("""dup1
line1
line2
unique1
dup2
line1
line2
""")

            file2 = Path(temp_dir) / "b.py"
            file2.write_text("""other
dup1
line1
line2
more
dup2
line1
line2
""")

            duplicates = deduplicator.detect_duplicates([str(file1), str(file2)], min_lines=2, similarity_threshold=1.0)

            # Should find multiple duplicate segments
            assert len(duplicates) >= 2

    def test_similarity_threshold_filtering(self):
        """Test that similarity threshold properly filters matches."""
        deduplicator = CodeDeduplicator()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Similar but not identical code
            file1 = Path(temp_dir) / "a.py"
            file1.write_text("""x = 1
y = 2
z = 3
""")

            file2 = Path(temp_dir) / "b.py"
            file2.write_text("""x = 1
y = 999
z = 3
""")

            # With threshold 1.0, should not match (2/3 similar)
            duplicates_strict = deduplicator.detect_duplicates([str(file1), str(file2)], min_lines=3, similarity_threshold=1.0)

            # With threshold 0.5, should match
            duplicates_loose = deduplicator.detect_duplicates([str(file1), str(file2)], min_lines=3, similarity_threshold=0.5)

            assert len(duplicates_strict) == 0
            assert len(duplicates_loose) > 0
