import unittest
import sys
import io
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch
from aura_cli.doctor import check_env_vars, check_embedding_index, main as aura_doctor_main
from aura_cli.commands import _handle_doctor


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
        lines = output.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("- ") and "Overall Health" not in line:
                # Example: - Python Version: PASS - Python version: 3.9.5
                # Remove '- ' prefix
                content = line[2:].strip()

                # Split by the first colon to separate check_name from the rest
                first_colon_idx = content.find(":")
                if first_colon_idx != -1:
                    check_name = content[:first_colon_idx].strip()
                    status_message_combined = content[first_colon_idx + 1 :].strip()

                    # Now split status_message_combined by the first hyphen to get status and message
                    first_hyphen_idx = status_message_combined.find("-")
                    if first_hyphen_idx != -1:
                        status = status_message_combined[:first_hyphen_idx].strip()
                        message = status_message_combined[first_hyphen_idx + 1 :].strip()
                        results[check_name] = {"status": status, "message": message}
                    else:
                        # Handle cases where there might be no hyphen, just a status
                        results[check_name] = {"status": status_message_combined, "message": ""}

        overall_health_line = [line for line in lines if "Overall Health:" in line]
        if overall_health_line:
            overall_health_status = overall_health_line[0].split("Overall Health:")[1].strip()
            results["Overall Health"] = {"status": overall_health_status.strip(), "message": ""}

        return results

    @patch("aura_cli.doctor.check_python_version")
    @patch("aura_cli.doctor.check_env_vars")
    @patch("aura_cli.doctor.check_sqlite_write_access")
    @patch("aura_cli.doctor.check_git_status")
    @patch("aura_cli.doctor.check_pytest_and_run_tests")
    def test_all_pass_scenario(self, mock_pytest, mock_git, mock_sqlite, mock_env, mock_python):
        # Mock sys.argv for this test
        original_argv = sys.argv
        sys.argv = ["aura_cli.doctor.py"]
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
            self.assertEqual(parsed_output["Pytest Tests"]["status"], "WARN")  # WARN because --run-tests not passed
            # Overall should be WARN because of Pytest
            self.assertEqual(parsed_output["Overall Health"]["status"], "WARN")
        finally:
            sys.argv = original_argv  # Restore original argv

    @patch("aura_cli.doctor.check_python_version")
    @patch("aura_cli.doctor.check_env_vars")
    @patch("aura_cli.doctor.check_sqlite_write_access")
    @patch("aura_cli.doctor.check_git_status")
    @patch("aura_cli.doctor.check_pytest_and_run_tests")
    def test_fail_scenario(self, mock_pytest, mock_git, mock_sqlite, mock_env, mock_python):
        # Mock sys.argv for this test
        original_argv = sys.argv
        sys.argv = ["aura_cli.doctor.py"]
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
            self.assertEqual(parsed_output["Overall Health"]["status"], "FAIL")  # Overall should be FAIL because of SQLite
        finally:
            sys.argv = original_argv  # Restore original argv

    @patch("aura_cli.doctor.check_python_version")
    @patch("aura_cli.doctor.check_env_vars")
    @patch("aura_cli.doctor.check_sqlite_write_access")
    @patch("aura_cli.doctor.check_git_status")
    @patch("aura_cli.doctor.check_pytest_and_run_tests")
    def test_all_pass_with_run_tests_scenario(self, mock_pytest, mock_git, mock_sqlite, mock_env, mock_python):
        # Simulate sys.argv to include --run-tests and --openrouter-api-key
        original_argv = sys.argv
        sys.argv = ["aura_cli.doctor.py", "--run-tests", "--openrouter-api-key", "dummy-key"]
        try:
            mock_python.return_value = ("PASS", "Python version: 3.9.5")
            mock_env.return_value = ("PASS", "OPENROUTER_API_KEY: Present")
            mock_sqlite.return_value = ("PASS", "SQLite write access: OK")
            mock_git.return_value = ("PASS", "Git is installed and repository is initialized.")
            mock_pytest.return_value = ("PASS", "Pytest tests passed.")  # Now it passes

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
            sys.argv = original_argv  # Restore original argv

    @patch.dict("os.environ", {"OPENAI_API_KEY": "openai-key"}, clear=True)
    def test_check_env_vars_reports_openai_only_as_pass(self):
        status, message = check_env_vars()

        self.assertEqual(status, "PASS")
        self.assertIn("OPENAI_API_KEY", message)
        self.assertIn("embeddings: OPENAI_API_KEY", message)

    @patch.dict("os.environ", {"AURA_LOCAL_MODEL_COMMAND": "ollama run llama2"}, clear=True)
    def test_check_env_vars_reports_local_only_as_warn_for_embeddings(self):
        status, message = check_env_vars()

        self.assertEqual(status, "PASS")
        self.assertIn("AURA_LOCAL_MODEL_COMMAND", message)
        self.assertIn("embeddings: disabled", message)

    def test_check_embedding_index_warns_when_active_model_has_no_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            db_path = root / "memory" / "brain.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.execute("CREATE TABLE memory_records (id TEXT PRIMARY KEY, content TEXT)")
            conn.execute("CREATE TABLE embeddings (record_id TEXT NOT NULL, model_id TEXT NOT NULL, dims INTEGER NOT NULL, data BLOB NOT NULL)")
            conn.execute("INSERT INTO memory_records (id, content) VALUES (?, ?)", ("rec1", "hello"))
            conn.execute(
                "INSERT INTO embeddings (record_id, model_id, dims, data) VALUES (?, ?, ?, ?)",
                ("rec1", "text-embedding-3-small", 3, b"123"),
            )
            conn.commit()
            conn.close()

            cfg = {
                "brain_db_path": "memory/brain.db",
                "semantic_memory": {"embedding_model": "local_profile:android_embeddings"},
                "local_model_profiles": {
                    "android_embeddings": {
                        "provider": "openai_compatible",
                        "embedding_model": "bge-small-en-v1.5-q8_0",
                    }
                },
            }

            status, detail = check_embedding_index(root, cfg)

        self.assertEqual(status, "WARN")
        self.assertIn("bge-small-en-v1.5-q8_0", detail)
        self.assertIn("text-embedding-3-small", detail)
        self.assertIn("memory reindex", detail)

    def test_check_embedding_index_passes_when_active_model_is_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            db_path = root / "memory" / "brain.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.execute("CREATE TABLE memory_records (id TEXT PRIMARY KEY, content TEXT)")
            conn.execute("CREATE TABLE embeddings (record_id TEXT NOT NULL, model_id TEXT NOT NULL, dims INTEGER NOT NULL, data BLOB NOT NULL)")
            conn.execute("INSERT INTO memory_records (id, content) VALUES (?, ?)", ("rec1", "hello"))
            conn.execute(
                "INSERT INTO embeddings (record_id, model_id, dims, data) VALUES (?, ?, ?, ?)",
                ("rec1", "bge-small-en-v1.5-q8_0", 3, b"123"),
            )
            conn.commit()
            conn.close()

            cfg = {
                "brain_db_path": "memory/brain.db",
                "semantic_memory": {"embedding_model": "local_profile:android_embeddings"},
                "local_model_profiles": {
                    "android_embeddings": {
                        "provider": "openai_compatible",
                        "embedding_model": "bge-small-en-v1.5-q8_0",
                    }
                },
            }

            status, detail = check_embedding_index(root, cfg)

        self.assertEqual(status, "PASS")
        self.assertIn("bge-small-en-v1.5-q8_0", detail)

    @patch("aura_cli.commands.capability_doctor_check", return_value=("PASS", "matched: docker_analysis"))
    @patch("aura_cli.doctor.check_pytest_and_run_tests", return_value=("WARN", "tests skipped"))
    @patch("aura_cli.doctor.check_git_status", return_value=("PASS", "git ok"))
    @patch("aura_cli.doctor.check_sqlite_write_access", return_value=("PASS", "sqlite ok"))
    @patch("aura_cli.doctor.check_env_vars", return_value=("PASS", "env ok"))
    @patch("aura_cli.doctor.check_python_version", return_value=("PASS", "python ok"))
    def test_handle_doctor_includes_capability_bootstrap_line(
        self,
        _mock_python,
        _mock_env,
        _mock_sqlite,
        _mock_git,
        _mock_pytest,
        _mock_capability,
    ):
        out = io.StringIO()
        with patch("sys.stdout", out):
            _handle_doctor()

        output = out.getvalue()
        self.assertIn("Capability", output)
        self.assertIn("matched: docker_analysis", output)


if __name__ == "__main__":
    unittest.main()
