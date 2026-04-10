"""
Unit tests for core/verification.py

Tests for verification framework, test execution, and coverage tracking.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, List

from core.verification import VerificationStatus, VerificationResult, Verifier, create_result


class TestVerificationStatus:
    """Tests for VerificationStatus enum."""

    def test_status_values(self):
        """Test enum values are correct."""
        assert VerificationStatus.PASS.value == "pass"
        assert VerificationStatus.FAIL.value == "fail"
        assert VerificationStatus.ERROR.value == "error"

    def test_status_comparison(self):
        """Test status can be compared."""
        assert VerificationStatus.PASS != VerificationStatus.FAIL
        assert VerificationStatus.PASS == VerificationStatus.PASS


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_result_creation(self):
        """Test creating a VerificationResult."""
        result = VerificationResult(status=VerificationStatus.PASS, message="Test passed", details={"coverage": 0.95})

        assert result.status == VerificationStatus.PASS
        assert result.message == "Test passed"
        assert result.details == {"coverage": 0.95}

    def test_result_with_none_details(self):
        """Test creating result with None details."""
        result = VerificationResult(status=VerificationStatus.FAIL, message="Test failed", details=None)

        assert result.details is None


class TestCreateResult:
    """Tests for create_result helper function."""

    def test_create_pass_result(self):
        """Test creating a pass result."""
        result = create_result(status=VerificationStatus.PASS, message="All good", details={"tests": 10})

        assert isinstance(result, VerificationResult)
        assert result.status == VerificationStatus.PASS
        assert result.message == "All good"
        assert result.details == {"tests": 10}

    def test_create_result_without_details(self):
        """Test creating result without details."""
        result = create_result(status=VerificationStatus.FAIL, message="Failed")

        assert result.details is None


class TestVerifier:
    """Tests for Verifier class."""

    @pytest.fixture
    def verifier(self):
        """Fixture providing a fresh Verifier instance."""
        return Verifier()

    def test_initialization(self, verifier):
        """Test Verifier initializes correctly."""
        assert verifier.test_cases == []
        assert verifier.failure_modes == {}
        assert verifier.coverage_data == {}

    def test_add_test_case_with_name(self, verifier):
        """Test adding a test case with explicit name."""
        test_func = lambda x: True

        verifier.add_test_case(test_func, "custom_name")

        assert len(verifier.test_cases) == 1
        assert verifier.test_cases[0][0] == test_func
        assert verifier.test_cases[0][1] == "custom_name"

    def test_add_test_case_without_name(self, verifier):
        """Test adding a test case without name uses function name."""

        def my_test_func(x):
            return True

        verifier.add_test_case(my_test_func)

        assert len(verifier.test_cases) == 1
        assert verifier.test_cases[0][1] == "my_test_func"

    def test_add_multiple_test_cases(self, verifier):
        """Test adding multiple test cases."""

        def test1(x):
            return True

        def test2(x):
            return False

        verifier.add_test_case(test1)
        verifier.add_test_case(test2, "custom_test2")

        assert len(verifier.test_cases) == 2

    def test_execute_test_success(self, verifier):
        """Test executing a passing test."""
        test_func = Mock(return_value=True)

        success, error = verifier._execute_test(test_func, "test_name", "code")

        assert success is True
        assert error == ""
        test_func.assert_called_once_with("code")

    def test_execute_test_failure(self, verifier):
        """Test executing a failing test."""
        test_func = Mock(return_value=False)

        success, error = verifier._execute_test(test_func, "test_name", "code")

        assert success is False
        assert "test_name failed" in error

    def test_execute_test_exception(self, verifier):
        """Test executing a test that raises exception."""

        def failing_test(code):
            raise ValueError("Test error")

        success, error = verifier._execute_test(failing_test, "test_name", "code")

        assert success is False
        assert "test_name:" in error
        assert "Test error" in error

    def test_predict_failure_modes_no_handling(self, verifier):
        """Test failure mode prediction for code without error handling."""
        code = "x = 1 + 1"

        modes = verifier.predict_failure_modes(code)

        assert "No error handling" in modes

    def test_predict_failure_modes_with_try(self, verifier):
        """Test failure mode prediction for code with error handling."""
        code = """
try:
    x = 1 / 0
except:
    pass
"""

        modes = verifier.predict_failure_modes(code)

        assert "No error handling" not in modes

    def test_predict_failure_modes_no_assertions(self, verifier):
        """Test failure mode prediction for code without assertions."""
        code = "x = 1 + 1"

        modes = verifier.predict_failure_modes(code)

        assert "No assertions" in modes

    def test_predict_failure_modes_with_assert(self, verifier):
        """Test failure mode prediction for code with assertions."""
        code = "assert x == 1"

        modes = verifier.predict_failure_modes(code)

        assert "No assertions" not in modes

    def test_predict_failure_modes_empty_code(self, verifier):
        """Test failure mode prediction for empty code."""
        modes = verifier.predict_failure_modes("")

        assert "No error handling" in modes
        assert "No assertions" in modes

    def test_track_coverage(self, verifier):
        """Test tracking coverage for a test."""
        verifier.track_coverage("test1", [1, 2, 3, 4, 5])

        assert "test1" in verifier.coverage_data
        assert verifier.coverage_data["test1"] == [1, 2, 3, 4, 5]

    def test_track_coverage_multiple_tests(self, verifier):
        """Test tracking coverage for multiple tests."""
        verifier.track_coverage("test1", [1, 2, 3])
        verifier.track_coverage("test2", [4, 5, 6])

        assert len(verifier.coverage_data) == 2
        assert verifier.coverage_data["test1"] == [1, 2, 3]
        assert verifier.coverage_data["test2"] == [4, 5, 6]

    def test_track_coverage_overwrite(self, verifier):
        """Test that tracking coverage overwrites existing data."""
        verifier.track_coverage("test1", [1, 2, 3])
        verifier.track_coverage("test1", [4, 5, 6])

        assert verifier.coverage_data["test1"] == [4, 5, 6]

    def test_calculate_coverage_stats_empty(self, verifier):
        """Test calculating coverage stats with no data."""
        stats = verifier.calculate_coverage_stats()

        assert stats["total_lines"] == 0
        assert stats["coverage_by_test"] == {}

    def test_calculate_coverage_stats_single_test(self, verifier):
        """Test calculating coverage stats with single test."""
        verifier.track_coverage("test1", [1, 2, 3, 4, 5])

        stats = verifier.calculate_coverage_stats()

        assert stats["total_lines"] == 5
        assert stats["coverage_by_test"] == {"test1": 5}

    def test_calculate_coverage_stats_multiple_tests(self, verifier):
        """Test calculating coverage stats with multiple tests."""
        verifier.track_coverage("test1", [1, 2, 3])
        verifier.track_coverage("test2", [3, 4, 5])  # Line 3 overlaps

        stats = verifier.calculate_coverage_stats()

        assert stats["total_lines"] == 5  # Unique lines: 1,2,3,4,5
        assert stats["coverage_by_test"]["test1"] == 3
        assert stats["coverage_by_test"]["test2"] == 3

    def test_calculate_coverage_stats_overlapping_lines(self, verifier):
        """Test coverage calculation with heavily overlapping lines."""
        verifier.track_coverage("test1", [1, 2, 3, 4, 5])
        verifier.track_coverage("test2", [1, 2, 3, 4, 5])  # Same lines

        stats = verifier.calculate_coverage_stats()

        assert stats["total_lines"] == 5  # Should not double count

    def test_get_verification_metrics_empty(self, verifier):
        """Test getting metrics with no tests run."""
        metrics = verifier.get_verification_metrics()

        assert metrics["num_tests"] == 0
        assert metrics["failure_modes"] == 0
        assert "coverage" in metrics

    def test_get_verification_metrics_with_data(self, verifier):
        """Test getting metrics with test data."""

        def test1(x):
            return True

        verifier.add_test_case(test1)
        verifier.track_coverage("test1", [1, 2, 3])

        metrics = verifier.get_verification_metrics()

        assert metrics["num_tests"] == 1
        assert metrics["coverage"]["total_lines"] == 3

    def test_run_verification_no_tests(self, verifier):
        """Test running verification with no test cases."""
        result = verifier.run_verification("some_code")

        assert isinstance(result, VerificationResult)
        assert result.status == VerificationStatus.PASS
        assert "All verifications passed" in result.message

    def test_run_verification_all_pass(self, verifier):
        """Test running verification with all tests passing."""

        def test1(code):
            return True

        def test2(code):
            return True

        verifier.add_test_case(test1)
        verifier.add_test_case(test2)

        result = verifier.run_verification("code")

        assert result.status == VerificationStatus.PASS
        assert "All verifications passed" in result.message
        assert "failure_modes" in result.details
        assert "coverage" in result.details

    def test_run_verification_with_failures(self, verifier):
        """Test running verification with failing tests."""

        def test1(code):
            return True

        def test2(code):
            return False

        verifier.add_test_case(test1)
        verifier.add_test_case(test2)

        result = verifier.run_verification("code")

        assert result.status == VerificationStatus.FAIL
        assert "Test failures detected" in result.message
        assert "failures" in result.details
        assert len(result.details["failures"]) == 1

    def test_run_verification_with_exception(self, verifier):
        """Test running verification when test raises exception."""

        def failing_test(code):
            raise RuntimeError("Test exception")

        verifier.add_test_case(failing_test)

        result = verifier.run_verification("code")

        assert result.status == VerificationStatus.FAIL
        assert len(result.details["failures"]) == 1
        assert "Test exception" in result.details["failures"][0]

    def test_run_verification_error_handling(self, verifier):
        """Test that run_verification handles its own errors."""
        # Mock calculate_coverage_stats to raise exception
        verifier.calculate_coverage_stats = Mock(side_effect=Exception("Coverage error"))

        result = verifier.run_verification("code")

        assert result.status == VerificationStatus.ERROR
        assert "Verification error" in result.message

    def test_run_verification_failure_modes_in_details(self, verifier):
        """Test that failure modes are included in result details."""
        code = "x = 1"  # No error handling, no assertions

        def test1(code):
            return True

        verifier.add_test_case(test1)

        result = verifier.run_verification(code)

        assert "failure_modes" in result.details
        assert "No error handling" in result.details["failure_modes"]
        assert "No assertions" in result.details["failure_modes"]


class TestIntegration:
    """Integration tests for verification module."""

    def test_full_verification_workflow(self):
        """Test complete verification workflow."""
        verifier = Verifier()

        # Add test cases
        def has_syntax(code):
            try:
                compile(code, "<string>", "exec")
                return True
            except SyntaxError:
                return False

        def has_functions(code):
            return "def " in code

        verifier.add_test_case(has_syntax, "syntax_check")
        verifier.add_test_case(has_functions, "function_check")

        # Test with valid code
        valid_code = """
def hello():
    return "world"
"""
        result = verifier.run_verification(valid_code)

        assert result.status == VerificationStatus.PASS

        # Track coverage
        verifier.track_coverage("syntax_check", [1, 2, 3, 4])
        verifier.track_coverage("function_check", [2, 3, 4])

        # Get metrics
        metrics = verifier.get_verification_metrics()
        assert metrics["num_tests"] == 2
        assert metrics["coverage"]["total_lines"] == 4

    def test_verification_with_complex_code(self):
        """Test verification with more complex code scenarios."""
        verifier = Verifier()

        def check_error_handling(code):
            return "try:" in code or "except" in code

        def check_assertions(code):
            return "assert" in code

        verifier.add_test_case(check_error_handling, "error_handling_check")
        verifier.add_test_case(check_assertions, "assertion_check")

        # Code without error handling or assertions
        bad_code = "x = 1 + 1"
        result = verifier.run_verification(bad_code)

        assert result.status == VerificationStatus.FAIL
        assert len(result.details["failures"]) == 2

        # Code with both
        good_code = """
try:
    x = 1 / 0
except ZeroDivisionError:
    assert False, "Should not reach here"
"""
        result = verifier.run_verification(good_code)

        assert result.status == VerificationStatus.PASS
