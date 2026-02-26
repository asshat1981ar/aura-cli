import unittest
import sys
import io
from unittest.mock import patch
from aura_cli.doctor import main as aura_doctor_main

class TestAuraDoctorOutputParsing(unittest.TestCase):

    def setUp(self):
        # Store original stdout
        self._original_stdout = sys.stdout
        # Redirect stdout to capture output
        sys.stdout = io.StringIO()

    def tearDown(self):
        # Restore original stdout
        sys.stdout = self._original_stdout

    def get_captured_output(self):
        return sys.stdout.getvalue()

    def parse_doctor_output(self, output: str):
        """
        Parses the output of aura_cli.doctor.py into a dictionary of checks and their status/message.
        """
        results = {}
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('- ') and "Overall Health" not in line:
                # Example: - Python Version: PASS - Python version: 3.9.5
                # Remove '- ' prefix
                content = line[2:].strip()
                
                # Split by the first colon to separate check_name from the rest
                first_colon_idx = content.find(':')
                if first_colon_idx != -1:
                    check_name = content[:first_colon_idx].strip()
                    status_message_combined = content[first_colon_idx + 1:].strip()

                    # Now split status_message_combined by the first hyphen to get status and message
                    first_hyphen_idx = status_message_combined.find('-')
                    if first_hyphen_idx != -1:
                        status = status_message_combined[:first_hyphen_idx].strip()
                        message = status_message_combined[first_hyphen_idx + 1:].strip()
                        results[check_name] = {"status": status, "message": message}
                    else:
                        # Handle cases where there might be no hyphen, just a status
                        results[check_name] = {"status": status_message_combined, "message": ""}
        
        overall_health_line = [line for line in lines if "Overall Health:" in line]
        if overall_health_line:
            overall_health_status = overall_health_line[0].split("Overall Health:")[1].strip()
            results["Overall Health"] = {"status": overall_health_status.strip(), "message": ""}
        
        return results

    @patch('aura_cli.doctor.check_python_version')
    @patch('aura_cli.doctor.check_env_vars')
    @patch('aura_cli.doctor.check_sqlite_write_access')
    @patch('aura_cli.doctor.check_git_status')
    @patch('aura_cli.doctor.check_pytest_and_run_tests')
    def test_all_pass_scenario(self, mock_pytest, mock_git, mock_sqlite, mock_env, mock_python):
        # Mock sys.argv for this test
        original_argv = sys.argv
        sys.argv = ['aura_cli.doctor.py']
        try:
            mock_python.return_value = ("PASS", "Python version: 3.9.5")
            mock_env.return_value = ("PASS", "OPENROUTER_API_KEY: Present")
            mock_sqlite.return_value = ("PASS", "SQLite write access: OK")
            mock_git.return_value = ("PASS", "Git is installed and repository is initialized.")
            mock_pytest.return_value = ("WARN", "Pytest is available, but tests were not run (use --run-tests).")
        
            aura_doctor_main()
            output = self.get_captured_output()
            parsed_output = self.parse_doctor_output(output)

            self.assertEqual(parsed_output["Python Version"]["status"], "PASS")
            self.assertEqual(parsed_output["Environment Variables"]["status"], "PASS")
            self.assertEqual(parsed_output["SQLite Write Access"]["status"], "PASS")
            self.assertEqual(parsed_output["Git Status"]["status"], "PASS")
            self.assertEqual(parsed_output["Pytest Tests"]["status"], "WARN") # WARN because --run-tests not passed
            # Overall should be WARN because of Pytest
            self.assertEqual(parsed_output["Overall Health"]["status"], "WARN") 
        finally:
            sys.argv = original_argv # Restore original argv

    @patch('aura_cli.doctor.check_python_version')
    @patch('aura_cli.doctor.check_env_vars')
    @patch('aura_cli.doctor.check_sqlite_write_access')
    @patch('aura_cli.doctor.check_git_status')
    @patch('aura_cli.doctor.check_pytest_and_run_tests')
    def test_fail_scenario(self, mock_pytest, mock_git, mock_sqlite, mock_env, mock_python):
        # Mock sys.argv for this test
        original_argv = sys.argv
        sys.argv = ['aura_cli.doctor.py']
        try:
            mock_python.return_value = ("PASS", "Python version: 3.9.5")
            mock_env.return_value = ("WARN", "OPENROUTER_API_KEY: Not found")
            mock_sqlite.return_value = ("FAIL", "SQLite write access: Failed (Permission denied)")
            mock_git.return_value = ("PASS", "Git is installed and repository is initialized.")
            mock_pytest.return_value = ("WARN", "Pytest is not installed.")

            aura_doctor_main()
            output = self.get_captured_output()
            parsed_output = self.parse_doctor_output(output)
            
            self.assertEqual(parsed_output["Python Version"]["status"], "PASS")
            self.assertEqual(parsed_output["Environment Variables"]["status"], "WARN")
            self.assertEqual(parsed_output["SQLite Write Access"]["status"], "FAIL")
            self.assertEqual(parsed_output["Git Status"]["status"], "PASS")
            self.assertEqual(parsed_output["Pytest Tests"]["status"], "WARN")
            self.assertEqual(parsed_output["Overall Health"]["status"], "FAIL") # Overall should be FAIL because of SQLite
        finally:
            sys.argv = original_argv # Restore original argv

    @patch('aura_cli.doctor.check_python_version')
    @patch('aura_cli.doctor.check_env_vars')
    @patch('aura_cli.doctor.check_sqlite_write_access')
    @patch('aura_cli.doctor.check_git_status')
    @patch('aura_cli.doctor.check_pytest_and_run_tests')
    def test_all_pass_with_run_tests_scenario(self, mock_pytest, mock_git, mock_sqlite, mock_env, mock_python):
        # Simulate sys.argv to include --run-tests and --openrouter-api-key
        original_argv = sys.argv
        sys.argv = ['aura_cli.doctor.py', '--run-tests', '--openrouter-api-key', 'dummy-key']
        try:
            mock_python.return_value = ("PASS", "Python version: 3.9.5")
            mock_env.return_value = ("PASS", "OPENROUTER_API_KEY: Present")
            mock_sqlite.return_value = ("PASS", "SQLite write access: OK")
            mock_git.return_value = ("PASS", "Git is installed and repository is initialized.")
            mock_pytest.return_value = ("PASS", "Pytest tests passed.") # Now it passes

            aura_doctor_main()
            output = self.get_captured_output()
            parsed_output = self.parse_doctor_output(output)

            self.assertEqual(parsed_output["Python Version"]["status"], "PASS")
            self.assertEqual(parsed_output["Environment Variables"]["status"], "PASS")
            self.assertEqual(parsed_output["SQLite Write Access"]["status"], "PASS")
            self.assertEqual(parsed_output["Git Status"]["status"], "PASS")
            self.assertEqual(parsed_output["Pytest Tests"]["status"], "PASS")
            self.assertEqual(parsed_output["Overall Health"]["status"], "PASS")
        finally:
            sys.argv = original_argv # Restore original argv

if __name__ == '__main__':
    unittest.main()
