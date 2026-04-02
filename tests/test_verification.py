"""Tests for core/verification.py."""

import pytest

from core.verification import (
    VerificationStatus,
    VerificationResult,
    create_result,
    Verifier,
)


class TestVerificationStatus:
    """Test VerificationStatus enum."""
    
    def test_status_values(self):
        """Test status enum values."""
        assert VerificationStatus.PASS.value == 'pass'
        assert VerificationStatus.FAIL.value == 'fail'
        assert VerificationStatus.ERROR.value == 'error'


class TestCreateResult:
    """Test create_result helper function."""
    
    def test_create_pass_result(self):
        """Test creating a pass result."""
        result = create_result(
            status=VerificationStatus.PASS,
            message="All tests passed"
        )
        assert result.status == VerificationStatus.PASS
        assert result.message == "All tests passed"
        assert result.details is None
    
    def test_create_result_with_details(self):
        """Test creating result with details."""
        details = {"coverage": 85, "tests_run": 10}
        result = create_result(
            status=VerificationStatus.FAIL,
            message="Some tests failed",
            details=details
        )
        assert result.details == details


class TestVerifier:
    """Test Verifier class."""
    
    def test_init(self):
        """Test Verifier initialization."""
        verifier = Verifier()
        assert verifier.test_cases == []
        assert verifier.failure_modes == {}
        assert verifier.coverage_data == {}
    
    def test_add_test_case(self):
        """Test adding test cases."""
        verifier = Verifier()
        
        def test_func(code):
            return True
        
        verifier.add_test_case(test_func, "my_test")
        assert len(verifier.test_cases) == 1
        assert verifier.test_cases[0] == (test_func, "my_test")
    
    def test_add_test_case_without_name(self):
        """Test adding test case without explicit name."""
        verifier = Verifier()
        
        def test_func(code):
            return True
        
        verifier.add_test_case(test_func)
        assert verifier.test_cases[0][1] == "test_func"
    
    def test_execute_test_success(self):
        """Test executing successful test."""
        verifier = Verifier()
        
        def passing_test(code):
            return True
        
        success, error = verifier._execute_test(passing_test, "passing_test", "code")
        assert success is True
        assert error == ''
    
    def test_execute_test_failure(self):
        """Test executing failing test."""
        verifier = Verifier()
        
        def failing_test(code):
            return False
        
        success, error = verifier._execute_test(failing_test, "failing_test", "code")
        assert success is False
        assert error == 'failing_test failed'
    
    def test_execute_test_exception(self):
        """Test executing test that raises exception."""
        verifier = Verifier()
        
        def error_test(code):
            raise ValueError("Test error")
        
        success, error = verifier._execute_test(error_test, "error_test", "code")
        assert success is False
        assert "Test error" in error
    
    def test_predict_failure_modes_no_error_handling(self):
        """Test failure mode detection for missing error handling."""
        verifier = Verifier()
        code = "x = 1 + 1"
        
        modes = verifier.predict_failure_modes(code)
        assert 'No error handling' in modes
    
    def test_predict_failure_modes_no_assertions(self):
        """Test failure mode detection for missing assertions."""
        verifier = Verifier()
        code = "x = 1 + 1"
        
        modes = verifier.predict_failure_modes(code)
        assert 'No assertions' in modes
    
    def test_predict_failure_modes_with_try_assert(self):
        """Test failure mode detection with try and assert present."""
        verifier = Verifier()
        code = """
try:
    x = 1 + 1
    assert x == 2
except:
    pass
"""
        modes = verifier.predict_failure_modes(code)
        assert 'No error handling' not in modes
        assert 'No assertions' not in modes
    
    def test_track_coverage(self):
        """Test tracking coverage data."""
        verifier = Verifier()
        verifier.track_coverage("test1", [1, 2, 3])
        verifier.track_coverage("test2", [3, 4, 5])
        
        assert verifier.coverage_data["test1"] == [1, 2, 3]
        assert verifier.coverage_data["test2"] == [3, 4, 5]
    
    def test_calculate_coverage_stats_empty(self):
        """Test coverage stats with no data."""
        verifier = Verifier()
        stats = verifier.calculate_coverage_stats()
        
        assert stats['total_lines'] == 0
        assert stats['coverage_by_test'] == {}
    
    def test_calculate_coverage_stats(self):
        """Test coverage stats calculation."""
        verifier = Verifier()
        verifier.track_coverage("test1", [1, 2, 3, 4])
        verifier.track_coverage("test2", [3, 4, 5, 6])
        
        stats = verifier.calculate_coverage_stats()
        
        assert stats['total_lines'] == 6  # Unique lines: 1,2,3,4,5,6
        assert stats['coverage_by_test'] == {'test1': 4, 'test2': 4}
    
    def test_get_verification_metrics(self):
        """Test getting verification metrics."""
        verifier = Verifier()
        
        def dummy_test(code):
            return True
        
        verifier.add_test_case(dummy_test, "test1")
        verifier.track_coverage("test1", [1, 2])
        
        metrics = verifier.get_verification_metrics()
        
        assert metrics['num_tests'] == 1
        assert metrics['failure_modes'] == 0
        assert 'coverage' in metrics
    
    def test_run_verification_all_pass(self):
        """Test verification with all passing tests."""
        verifier = Verifier()
        
        def passing_test(code):
            return True
        
        verifier.add_test_case(passing_test, "passing")
        
        result = verifier.run_verification("code")
        
        assert result.status == VerificationStatus.PASS
        assert "All verifications passed" in result.message
        assert 'failure_modes' in result.details
        assert 'coverage' in result.details
    
    def test_run_verification_with_failures(self):
        """Test verification with failing tests."""
        verifier = Verifier()
        
        def failing_test(code):
            return False
        
        verifier.add_test_case(failing_test, "failing")
        
        result = verifier.run_verification("code")
        
        assert result.status == VerificationStatus.FAIL
        assert "Test failures detected" in result.message
        assert 'failures' in result.details
    
    def test_run_verification_with_exception(self):
        """Test verification when test raises exception."""
        verifier = Verifier()
        
        def error_test(code):
            raise RuntimeError("Unexpected error")
        
        verifier.add_test_case(error_test, "error_test")
        
        result = verifier.run_verification("code")
        
        # Exception in test is treated as failure, not verification error
        assert result.status == VerificationStatus.FAIL
        assert "error_test: Unexpected error" in result.details['failures']
