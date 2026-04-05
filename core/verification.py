from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

class VerificationStatus(Enum):
    PASS = 'pass'
    FAIL = 'fail' 
    ERROR = 'error'

@dataclass
class VerificationResult:
    status: VerificationStatus
    message: str
    details: Dict = None
    
def create_result(status: VerificationStatus, message: str, details: Dict = None) -> VerificationResult:
    """Helper to create VerificationResult objects"""
    return VerificationResult(status=status, message=message, details=details)

class Verifier:
    def __init__(self):
        self.test_cases = []
        self.failure_modes = {}
        self.coverage_data = {}
    
    def add_test_case(self, test_func, name: str = None):
        """Add a test case with optional name"""
        self.test_cases.append((test_func, name or test_func.__name__))
    
    def _execute_test(self, test_func, test_name: str, code: str) -> Tuple[bool, str]:
        """Execute a single test case and return success flag and error message"""
        try:
            result = test_func(code)
            return result, '' if result else f'{test_name} failed'
        except Exception as e:
            return False, f'{test_name}: {str(e)}'
        
    def predict_failure_modes(self, code: str) -> List[str]:
        """Analyze code to predict potential failure modes"""
        modes = []
        if 'try:' not in code:
            modes.append('No error handling')
        if 'assert' not in code:
            modes.append('No assertions')
        return modes
        
    def track_coverage(self, test_name: str, covered_lines: List[int]):
        """Track code coverage for a test case"""
        self.coverage_data[test_name] = covered_lines
        
    def calculate_coverage_stats(self) -> Dict:
        """Calculate overall coverage statistics"""
        total_lines = set()
        for lines in self.coverage_data.values():
            total_lines.update(lines)
        return {
            'total_lines': len(total_lines),
            'coverage_by_test': {k: len(v) for k,v in self.coverage_data.items()}
        }
    
    def get_verification_metrics(self) -> Dict:
        """Get metrics about verification process"""
        return {
            'num_tests': len(self.test_cases),
            'failure_modes': len(self.failure_modes), 
            'coverage': self.calculate_coverage_stats()
        }
    
    def run_verification(self, code: str) -> VerificationResult:
        """Run all verification checks on code"""
        try:
            # Run test cases
            failures = []
            for test, name in self.test_cases:
                success, error = self._execute_test(test, name, code)
                if not success:
                    failures.append(error)
                    
            # Get failure modes and coverage
            modes = self.predict_failure_modes(code)
            coverage = self.calculate_coverage_stats()
            
            if failures:
                return create_result(
                    status=VerificationStatus.FAIL,
                    message='Test failures detected',
                    details={
                        'failures': failures,
                        'failure_modes': modes,
                        'coverage': coverage
                    }
                )
            
            return create_result(
                status=VerificationStatus.PASS,
                message='All verifications passed', 
                details={
                    'failure_modes': modes,
                    'coverage': coverage
                }
            )
            
        except Exception as e:
            return create_result(
                status=VerificationStatus.ERROR,
                message=f'Verification error: {str(e)}'
            )
